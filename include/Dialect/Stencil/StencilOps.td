#ifndef STENCIL_OPS
#define STENCIL_OPS

include "Dialect/Stencil/StencilBase.td"
include "Dialect/Stencil/StencilInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Concrete Operations
//===----------------------------------------------------------------------===//

def Stencil_CastOp : Stencil_Op<"cast", [DeclareOpInterfaceMethods<ShapeOp>]> {
  let summary = "cast the dynamiclly shaped input fields";
  let description = [{
    This operation casts dynamically shaped input fields to statically shaped fields.

    Example:
      %0 = stencil.cast %in ([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  }];

  let arguments = (ins Stencil_Field:$field, Stencil_Index:$lb, Stencil_Index:$ub);
  let results = (outs Stencil_Field:$res);

  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "Value field, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub", [{
      state.addOperands(field);
      state.addAttribute(getLBAttrName(), builder.getI64ArrayAttr(lb));
      state.addAttribute(getUBAttrName(), builder.getI64ArrayAttr(ub));
      state.addTypes(stencil::FieldType::get(
        field.getType().cast<stencil::GridType>().getElementType(), 
        applyFunElementWise(lb, ub, std::minus<int64_t>())));
    }]>
  ];

  let assemblyFormat = [{
    $field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` functional-type($field, $res)
  }];

  let verifier = [{
    auto fieldType = field().getType().cast<stencil::GridType>();
    auto resType = res().getType().cast<stencil::GridType>();
    auto shapeOp = cast<ShapeOp>(this->getOperation());
    if(!fieldType.hasDynamicShape())
      return emitOpError("expected field to have dynamic shape");
    if(resType.hasDynamicShape())
      return emitOpError("expected result to have static shape");
    if(fieldType.getAllocation() != resType.getAllocation())
      return emitOpError("expected the field and result types to have the same allocation");
    if (fieldType.getElementType() != resType.getElementType())
      return emitOpError("the field and result types have different element type");
    if(shapeOp.getRank() != fieldType.getRank() || shapeOp.getRank() != resType.getRank())
      return emitOpError("expected op and the field and result types to have the same rank");
    
    // Ensure the shape matches the result type
    if (llvm::any_of(
          llvm::zip(resType.getAllocation(),
                    resType.getShape(),
                    applyFunElementWise(shapeOp.getUB(), shapeOp.getLB(), std::minus<int64_t>())),
          [&](std::tuple<int64_t, int64_t, int64_t> x) { return std::get<0>(x) && (std::get<1>(x) != std::get<2>(x)); }))
      return emitOpError("expected op and result type to have the same shape");

    // Verify all users fit the shape
    for(auto user : res().getUsers()) {
      if(auto userOp = dyn_cast<ShapeOp>(user)) {
        if(userOp.hasShape() &&
           (shapeOp.getLB() != applyFunElementWise(shapeOp.getLB(), userOp.getLB(), min) ||
            shapeOp.getUB() != applyFunElementWise(shapeOp.getUB(), userOp.getUB(), max))) 
          return emitOpError("shape not large enough to fit all accesses");
      }
    }

    return success();
  }];
  
  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Stencil_IndexOp : Stencil_Op<"index", [DeclareOpInterfaceMethods<OffsetOp>,
    NoSideEffect]> {
  let summary = "stencil index operation";
  let description = [{
    This operation returns the index of the current loop iteration for the
    chosen direction (0, 1, or 2).
    The offset is specified relative to the current position.

    Example:
      %0 = stencil.index 0 [-1, 0, 0] : index
  }];

  let arguments = (ins Confined<I64Attr, [IntMinValue<0>, IntMaxValue<2>]>:$dim,
                     Stencil_Index:$offset);
  let results = (outs Index:$idx);

  let assemblyFormat = [{
    $dim $offset attr-dict-with-keyword `:` type($idx)
  }];

  let verifier = [{
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getOffsetAttrName() { return "offset"; }
    static StringRef getDimAttrName() { return "dim";}
    int64_t getDim() {
      return dim().getSExtValue();
    }
  }];
}

def Stencil_AccessOp : Stencil_Op<"access", [DeclareOpInterfaceMethods<OffsetOp>,
    NoSideEffect]> {
  let summary = "stencil access operation";
  let description = [{
    This operation accesses a temporary element given a constant 
    offset. The offset is specified relative to the current position.

    Example:
      %0 = stencil.access %temp [-1, 0, 0] : !stencil.temp<?x?x?xf64> -> f64
  }];

  let arguments = (ins Stencil_Temp:$temp, Stencil_Index:$offset);
  let results = (outs Stencil_ElementType:$res);
  
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "Value temp, ArrayRef<int64_t> offset", [{
      state.addOperands(temp);
      state.addAttribute(getOffsetAttrName(), builder.getI64ArrayAttr(offset));
      auto tempType = temp.getType().cast<stencil::GridType>();
      state.addTypes(tempType.getElementType());
    }]>
  ];

  let assemblyFormat = [{
    $temp $offset attr-dict-with-keyword `:` functional-type($temp, $res)
  }];

  let verifier = [{
    auto tempType = temp().getType().cast<stencil::GridType>();
    if (offset().size() != tempType.getRank())
      return emitOpError("offset and temp dimensions do not match");
    if (res().getType() != tempType.getElementType())
      return emitOpError("result type and element type are inconsistent");
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getOffsetAttrName() { return "offset"; }
  }];
}

def Stencil_DynAccessOp : Stencil_Op<"dyn_access", [NoSideEffect]> {
  let summary = "stencil dyn_access operation";
  let description = [{
    This operation accesses a temporary element given a dynamic offset. 
    The offset is specified in absolute coordinates. An additional 
    range attribute specifies the maximal access extent relative to the
    iteration domain of the parent apply operation. 

    Example:
      %0 = stencil.dyn_access %temp [%i, %j, %k] in [-1, -1, -1]:[1, 1, 1] : !stencil.temp<?x?x?xf64> -> f64
  }];

  let arguments = (ins Stencil_Temp:$temp, 
                       Variadic<Index>:$offset, 
                       Stencil_Index:$lb, 
                       Stencil_Index:$ub);
  let results = (outs Stencil_ElementType:$res);
  
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "Value temp, ValueRange offset, "
              "ArrayRef<int64_t> lb, ArrayRef<int64_t> ub", [{
      state.addOperands(temp);
      state.addOperands(offset);
      state.addAttribute(getLBAttrName(), builder.getI64ArrayAttr(lb));
      state.addAttribute(getUBAttrName(), builder.getI64ArrayAttr(ub));
      auto tempType = temp.getType().cast<stencil::GridType>();
      state.addTypes(tempType.getElementType());
    }]>
  ];

  let assemblyFormat = [{
    $temp `[` $offset `]` `in` $lb `:` $ub attr-dict-with-keyword `:` functional-type($temp, $res)
  }];

  let verifier = [{
    auto tempType = temp().getType().cast<stencil::GridType>();
    if (offset().size() != tempType.getRank())
      return emitOpError("offset and temp dimensions do not match");
    if (res().getType() != tempType.getElementType())
      return emitOpError("result type and element type are inconsistent");
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Stencil_DependOp : Stencil_Op<"depend", [DeclareOpInterfaceMethods<OffsetOp>,
    NoSideEffect]> {
  let summary = "stencil depend operation";
  let description = [{
    This operation accesses an output at a previously updated offset. The operation
    takes an index that specifies the output to access and an offset relative
    to the current position. The operation only works in an apply that implements
    a sequential loop and only offset in the negative loop direction are supported.
    
    Example:
      %0 = stencil.depend 0 [-1, 0, 0] :  f64
  }];

  let arguments = (ins I64Attr:$output, Stencil_Index:$offset);
  let results = (outs Stencil_ElementType:$res);
  
  let assemblyFormat = [{
    $output $offset attr-dict-with-keyword `:` type($res)
  }];

  let verifier = [{
    auto applyOp = getParentOfType<stencil::ApplyOp>();
    if(!applyOp || !applyOp.seq().hasValue())
      return emitOpError("expected parent that executes sequentially");
    
    // Check the output offset exists and the result type matches
    int64_t output = getOutput();
    if(!(output >= 0 && output < applyOp.getNumResults()))
      return emitOpError("output index out of range");
    auto resultType = applyOp.getResult(output).getType().cast<stencil::GridType>();
    if(resultType.getElementType() != res().getType())
      return emitOpError("result type does not match the output element type");
    
    // Check the offset
    auto offsetOp = cast<OffsetOp>(this->getOperation());
    for (auto &en : llvm::enumerate(offsetOp.getOffset())) {
      if(en.value() != 0) {
        if(applyOp.getSeqDim() != en.index())
          return emitOpError("unexpected non-zero offset in parallel dimension");
        if(applyOp.getSeqDir() == 1 && en.value() > -1)
          return emitOpError("expected offset in the negative direction");
        if(applyOp.getSeqDir() == -1 && en.value() < 1)
          return emitOpError("expected offset in the positive direction");
      }
    }
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getOffsetAttrName() { return "offset"; }
    int64_t getOutput() { return output().getSExtValue(); }
  }];
}

def Stencil_LoadOp : Stencil_Op<"load", [DeclareOpInterfaceMethods<ShapeOp>,
                                         NoSideEffect]> {
  let summary = "load operation";
  let description = [{
    This operation takes a field and returns a temporary values.

    Example:
      %0 = stencil.load %field : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  }];

  let arguments = (ins Stencil_Field:$field,
                       OptionalAttr<Stencil_Index>:$lb, 
                       OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Stencil_Temp:$res);
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Value field", [{
      state.addOperands(field);
      auto fieldType = field.getType().cast<stencil::GridType>();
      state.addTypes(stencil::TempType::get(fieldType.getElementType(), fieldType.getShape()));
    }]>
  ];

  let assemblyFormat = [{
    $field (`(` $lb^ `:` $ub `)`)? attr-dict-with-keyword `:` functional-type($field, $res)
  }];

  let verifier = [{
    // Check the field and result types
    auto fieldType = field().getType().cast<stencil::GridType>();
    auto resType = res().getType().cast<stencil::GridType>();
    if(fieldType.hasDynamicShape())
      return emitOpError("expected fields to have static shape");
    if (fieldType.getRank() != resType.getRank())
      return emitOpError("the field and temp types have different rank");
    if (fieldType.getAllocation() != resType.getAllocation())
      return emitOpError("the field and temp types have different allocation");
    if (fieldType.getElementType() != resType.getElementType())
      return emitOpError("the field and temp types have different element type");
    
    if(!isa<stencil::CastOp>(field().getDefiningOp()))
      return emitOpError("expected the defining op of the field is a cast operation");

    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Stencil_StoreOp : Stencil_Op<"store", [DeclareOpInterfaceMethods<ShapeOp>]> {
  let summary = "store operation";
  let description = [{
    This operation takes a temp and writes a field on a user defined range.

    Example:
      stencil.store %temp to %field ([0,0,0] : [64,64,60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  }];

  let arguments = (ins Stencil_Temp:$temp, 
                       Stencil_Field:$field, 
                       Stencil_Index:$lb, 
                       Stencil_Index:$ub);
  let results = (outs);
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Value temp, "
              "Value field, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub", [{
      state.addOperands({temp, field});
      state.addAttribute(getLBAttrName(), builder.getI64ArrayAttr(lb));
      state.addAttribute(getUBAttrName(), builder.getI64ArrayAttr(ub)); 
    }]>
  ];
  
  let assemblyFormat = [{
    $temp `to` $field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` type($temp) `to` type($field)
  }];

  let verifier = [{
    // Check the field and result types
    auto fieldType = field().getType().cast<stencil::GridType>();
    auto tempType = temp().getType().cast<stencil::GridType>();
    if(fieldType.hasDynamicShape())
      return emitOpError("expected fields to have static shape");
    if (fieldType.getRank() != tempType.getRank())
      return emitOpError("the field and temp types have different rank");
    if (fieldType.getRank() != tempType.getRank())
      return emitOpError("the field and temp types have different rank");
    if (fieldType.getAllocation() != tempType.getAllocation())
      return emitOpError("the field and temp types have different allocation");
    if (fieldType.getElementType() != tempType.getElementType())
      return emitOpError("the field and temp types have different element type");
    
    if (!dyn_cast<stencil::ApplyOp>(temp().getDefiningOp()))
      return emitOpError("output temp not result of an apply");
    if (llvm::count_if(field().getUsers(), [](Operation *op) { return isa_and_nonnull<stencil::LoadOp>(op); }) != 0)
      return emitOpError("an output cannot be an input");  
    if (llvm::count_if(field().getUsers(), [](Operation *op) { return isa_and_nonnull<stencil::StoreOp>(op); }) != 1)
      return emitOpError("multiple stores to the same output");  
    
    if(!isa<stencil::CastOp>(field().getDefiningOp()))
      return emitOpError("expected the defining op of the field is a cast operation");

    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Stencil_ApplyOp : Stencil_Op<"apply", [DeclareOpInterfaceMethods<ShapeOp, ["setOperandShape"]>,
                                           IsolatedFromAbove, 
                                           SingleBlockImplicitTerminator<"ReturnOp">,
                                           NoSideEffect]> {
  let summary = "apply operation";
  let description = [{
    This operation takes a stencil function plus parameters and applies 
    the stencil function to the output temp.

    Example:

      %0 = stencil.apply (%arg0=%0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
        ...
      } 
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                        OptionalAttr<Stencil_Index>:$lb, 
                        OptionalAttr<Stencil_Index>:$ub,
                        OptionalAttr<Stencil_Loop>:$seq);
  let results = (outs Variadic<Stencil_Temp>:$res);
  let regions = (region SizedRegion<1>:$region);
  let hasCanonicalizer = 1;
  
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "ValueRange operands, ValueRange results, "
              "Optional<ArrayAttr> seq", [{
      state.addOperands(operands);
      auto region = state.addRegion(); 
      region->push_back(new Block());
      for(auto operand : operands) {
        region->front().addArgument(operand.getType());
      }
      if(seq.hasValue()) {
        state.addAttribute(getSeqAttrName(), seq.getValue());
      }
      SmallVector<Type, 10> resultTypes;
      resultTypes.resize(results.size());
      llvm::transform(results, resultTypes.begin(),
                      [](Value value) { return value.getType(); });
      state.addTypes(resultTypes);
    }]>
  ];

  let verifier = [{
    // Check the operands
    if (region().front().getNumArguments() != operands().size())
      return emitOpError("operand and argument counts do not match");
    for (unsigned i = 0, e = operands().size(); i != e; ++i) {
      if (region().front().getArgument(i).getType() != operands()[i].getType())
        return emitOpError("operand and argument types do not match");
    }

    // Check the results
    auto shapeOp = cast<ShapeOp>(this->getOperation());
    for(auto result : res()) {
      auto tempType = result.getType().cast<GridType>();
      if(shapeOp.hasShape() && shapeOp.getRank() != tempType.getRank())
        return emitOpError("expected result rank to match the operation rank");
    }

    // Check the sequential loop attributes
    if(seq().hasValue()) {
      if(getSeqDim() < 0)
        return emitOpError("expected seqential loop dimension to be positive");
      if(getSeqDir() != 1 && getSeqDir() != -1)
        return emitOpError("expected sequential loop direction to be -1 or 1");
      if(shapeOp.hasShape()) {
        if(getSeqDim() >= shapeOp.getRank())
          return emitOpError("expected sequential loop dimension to match shape");
        if(getSeqLB() != shapeOp.getLB()[getSeqDim()] || getSeqUB() != shapeOp.getUB()[getSeqDim()])
          return emitOpError("expected sequential loop range to match shape");
      }
    }
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    static StringRef getSeqAttrName() { return "seq"; }
    int64_t getSeqDim() { return seq().getValue()[0].cast<IntegerAttr>().getValue().getSExtValue(); }
    int64_t getSeqLB() { return seq().getValue()[1].cast<IntegerAttr>().getValue().getSExtValue(); }
    int64_t getSeqUB() { return seq().getValue()[2].cast<IntegerAttr>().getValue().getSExtValue(); }
    int64_t getSeqDir() { return seq().getValue()[3].cast<IntegerAttr>().getValue().getSExtValue(); }
    Block *getBody() { return &region().front(); }
  }];
}

def Stencil_ReturnOp : Stencil_Op<"return", [Terminator,
                                             HasParent<"ApplyOp">]> {
  let summary = "return operation";
  let description = [{
    The return operation terminates the the stencil apply and writes
    the results of the stencil operator to the temporary values returned
    by the stencil apply operation. The types and the number of operands
    must match the results of the stencil apply operation.

    The optional unroll attribute enables the implementation of loop
    unrolling at the stencil dialect level.

    Examples:
      stencil.return %0 : f64
  }];

  let arguments = (ins Variadic<Stencil_ElementType>:$operands,
                   OptionalAttr<Stencil_Index>:$unroll);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "ValueRange operands, Optional<ArrayAttr> unroll", [{
      state.addOperands({operands});
      if(unroll.hasValue())
        state.addAttribute(getUnrollAttrName(), unroll.getValue());
    }]>
  ];

  let assemblyFormat = [{
    (`unroll` $unroll^)? $operands attr-dict-with-keyword `:` type($operands)
  }];

  let verifier = [{
    auto applyOp = cast<stencil::ApplyOp>(getParentOp());
    unsigned unrollFactor = getUnrollFactor();
    size_t unrollDim = getUnrollDimension();

    // Verify the size of the unroll dimension is a multiple of the unroll factor
    auto shapeOp = cast<ShapeOp>(applyOp.getOperation());
    if(shapeOp.hasShape() && (shapeOp.getUB()[unrollDim] - shapeOp.getLB()[unrollDim]) % unrollFactor != 0)
      return emitOpError("expected domain size to be a multiple of the unroll factor");

    // Verify the number of operands matches the number of apply results
    auto results = applyOp.res();
    if (getNumOperands() != unrollFactor * results.size())
      return emitOpError("the operand and apply result counts do not match");
  
    // Verify the element types match
    for (unsigned i = 0, e = results.size(); i != e; ++i) {
      auto tempType = applyOp.getResult(i).getType().cast<GridType>();
      for (unsigned j = 0; j < unrollFactor; j++)
        if (getOperand(i * unrollFactor + j).getType() != tempType.getElementType())
          return emitOpError("the operand and apply result element types do not match");
    }
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getUnrollAttrName() { return "unroll"; }
    Index getUnroll() {
      Index result;
      ArrayAttr unrollAttr = unroll().getValue();
      for (auto &elem : unrollAttr) 
        result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
      return result;
    }
    unsigned getUnrollFactor() {
      unsigned factor = 1;
      if (unroll().hasValue()) {
        Index unroll = getUnroll();
        factor = std::accumulate(unroll.begin(), unroll.end(), 1,
                                      std::multiplies<int64_t>());
      }
      return factor;
    }
    size_t getUnrollDimension() {
      size_t dimension = 0;
      if (unroll().hasValue()) {
        auto unroll = getUnroll();
        auto unrollFac = getUnrollFactor();
        auto it = llvm::find_if(unroll, [&](int64_t x) {
          return x == unrollFac;
        });
        assert(it != unroll.end() &&
              "expected to find unroll factor");
        dimension = std::distance(unroll.begin(), it);
      }
      return dimension;
    } 
  }];
}

#endif // STENCIL_OPS
