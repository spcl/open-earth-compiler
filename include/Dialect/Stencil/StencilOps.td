#ifndef STENCIL_OPS
#define STENCIL_OPS

include "Dialect/Stencil/StencilBase.td"
include "Dialect/Stencil/StencilInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Concrete Operations
//===----------------------------------------------------------------------===//

def Stencil_CastOp : Stencil_Op<"cast", [
  DeclareOpInterfaceMethods<ShapeOp>]> {
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

def Stencil_IndexOp : Stencil_Op<"index", [
    DeclareOpInterfaceMethods<OffsetOp>,
    DeclareOpInterfaceMethods<ShiftOp>,
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

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "int64_t dim, ArrayRef<int64_t> offset", [{
      state.addAttribute(getDimAttrName(), builder.getI64IntegerAttr(dim));
      state.addAttribute(getOffsetAttrName(), builder.getI64ArrayAttr(offset));
      state.addTypes(builder.getIndexType());
    }]>
  ];

  let assemblyFormat = [{
    $dim $offset attr-dict-with-keyword `:` type($idx)
  }];

  let verifier = [{
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getOffsetAttrName() { return "offset"; }
    static StringRef getDimAttrName() { return "dim";}
  }];
}

def Stencil_AccessOp : Stencil_Op<"access", [
    DeclareOpInterfaceMethods<OffsetOp>,
    DeclareOpInterfaceMethods<ShiftOp>,
    DeclareOpInterfaceMethods<ExtentOp>,
    NoSideEffect]> {
  let summary = "stencil access operation";
  let description = [{
    This operation accesses a temporary element given a constant 
    offset. The offset is specified relative to the current position.

    Example:
      %0 = stencil.access %temp [-1, 0, 0] : !stencil.temp<?x?x?xf64> -> f64
  }];

  let arguments = (ins Stencil_Temp:$temp, Stencil_Index:$offset);
  let results = (outs Stencil_Element:$res);
  
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

def Stencil_DynAccessOp : Stencil_Op<"dyn_access", [
  DeclareOpInterfaceMethods<ShiftOp, ["shiftByOffset"]>,
  DeclareOpInterfaceMethods<ExtentOp, ["getAccessExtent"]>,
  NoSideEffect]> {
  let summary = "stencil dyn_access operation";
  let description = [{
    This operation accesses a temporary element given a dynamic offset. 
    The offset is specified in absolute coordinates. An additional 
    range attribute specifies the maximal access extent relative to the
    iteration domain of the parent apply operation. 

    Example:
      %0 = stencil.dyn_access %temp (%i, %j, %k) in [-1, -1, -1] : [1, 1, 1] : !stencil.temp<?x?x?xf64> -> f64
  }];

  let arguments = (ins Stencil_Temp:$temp, 
                       Variadic<Index>:$offset, 
                       Stencil_Index:$lb, 
                       Stencil_Index:$ub);
  let results = (outs Stencil_Element:$res);
  
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
    $temp `(` $offset `)` `in` $lb `:` $ub attr-dict-with-keyword `:` functional-type($temp, $res)
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

def Stencil_LoadOp : Stencil_Op<"load", [
  DeclareOpInterfaceMethods<ShapeOp>,
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

def Stencil_BufferOp : Stencil_Op<"buffer", [DeclareOpInterfaceMethods<ShapeOp>]> {
  let summary = "buffer operation";
  let description = [{
    Prevents fusion of consecutive stencil.apply operations.

    Example:
      %0 = stencil.buffer %buffered : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  }];

  let arguments = (ins Stencil_Temp:$temp,
                       OptionalAttr<Stencil_Index>:$lb,
                       OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Stencil_Temp:$res);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Value temp", [{
      state.addOperands(temp);
      state.addTypes(temp.getType());
    }]>
  ];

  let assemblyFormat = [{
    $temp (`(` $lb^ `:` $ub`)`)? attr-dict-with-keyword `:` functional-type($temp, $res)
  }];

  let verifier = [{
    if(!isa<stencil::ApplyOp>(temp().getDefiningOp()))
      return emitOpError("expected input to buffer to be apply op");
    if(!temp().hasOneUse())
      return emitOpError("expected input to buffer to be only used once");
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Stencil_StoreOp : Stencil_Op<"store", [
  DeclareOpInterfaceMethods<ShapeOp>]> {
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
    
    if (!(dyn_cast<stencil::ApplyOp>(temp().getDefiningOp()) || 
          dyn_cast<stencil::CombineOp>(temp().getDefiningOp())))
      return emitOpError("output temp not result of an apply or a combine op");
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

def Stencil_ApplyOp : Stencil_Op<"apply", [
  DeclareOpInterfaceMethods<ShapeOp, ["updateArgumentTypes"]>,
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
                        OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Variadic<Stencil_Temp>:$res);
  let regions = (region SizedRegion<1>:$region);
  let hasCanonicalizer = 1;
  
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "TypeRange resultTypes, ValueRange operands, "
              "Optional<ArrayAttr> lb, Optional<ArrayAttr> ub", [{
      state.addOperands(operands);
      if(lb.hasValue() && ub.hasValue()) {
        state.addAttribute(getLBAttrName(), lb.getValue());
        state.addAttribute(getUBAttrName(), ub.getValue()); 
      }
      auto region = state.addRegion(); 
      region->push_back(new Block());
      for(auto operand : operands) {
        region->front().addArgument(operand.getType());
      }
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
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    Block *getBody() { return &region().front(); }
  }];
}

def Stencil_StoreResultOp : Stencil_Op<"store_result", []> {
  let summary = "store_result operation";
  let description = [{
    The store_result operation either stores an operand value or nothing.

    Examples:
      stencil.store_result %0 : !stencil.result<f64>
      stencil.store_result : !stencil.result<f64>      
  }];

  let arguments = (ins Variadic<Stencil_Element>:$operands);
  let results = (outs Stencil_Result:$res);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "Value value", [{
      state.addOperands({value});
      state.addTypes(stencil::ResultType::get(value.getType()));
    }]>
  ];

  let assemblyFormat = [{
    ($operands^)? attr-dict-with-keyword `:` functional-type($operands, $res)
  }];

  let verifier = [{
    // Check at most one operand
    if(operands().size() > 1)
      return emitOpError("expected at most one operand");

    // Check the return type
    auto resultType = res().getType().cast<stencil::ResultType>().getResultType();
    if(operands().size() == 1 && resultType != operands()[0].getType())
      return emitOpError("operand type and result type are inconsistent");
      
    // Check the result mapping
    if(!getReturnOpOperand())
      return emitOpError("result not mapped to a unique stencil.return operand");
    return success();
  }];

  let extraClassDeclaration = [{
    OpOperand* getReturnOpOperand(); 
  }];
}

def Stencil_ReturnOp : Stencil_Op<"return", [
  Terminator,
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
      stencil.return %0 : !stencil.result<f64>
  }];

  let arguments = (ins Variadic<Stencil_Result>:$operands,
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

    // Verify the number of operands matches the number of apply results
    auto results = applyOp.res();
    if (getNumOperands() != unrollFactor * results.size())
      return emitOpError("the operand and apply result counts do not match");
  
    // Verify the element types match
    for (unsigned i = 0, e = results.size(); i != e; ++i) {
      auto tempType = applyOp.getResult(i).getType().cast<GridType>();
      for (unsigned j = 0; j < unrollFactor; j++)
        if (getOperand(i * unrollFactor + j).getType().cast<stencil::ResultType>().getResultType() != tempType.getElementType())
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

def Stencil_CombineOp : Stencil_Op<"combine", [DeclareOpInterfaceMethods<ShapeOp>, SameVariadicOperandSize]> {
  let summary = "combine operation";
  let description = [{
    Prevents fusion of consecutive stencil.apply operations.

    Example:
      %result = stencil.combine 2 at 11 %lower, %upper : (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  }];

  let arguments = (ins Confined<I64Attr, [IntMinValue<0>, IntMaxValue<2>]>:$dim,
                       I64Attr:$index,
                       Variadic<Stencil_Temp>:$lower,
                       Variadic<Stencil_Temp>:$upper,
                       OptionalAttr<Stencil_Index>:$lb,
                       OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Variadic<Stencil_Temp>:$res);

  let assemblyFormat = [{
    $dim `at` $index `lower` `=` `(`$lower `:` type($lower)`)` `upper` `=` `(`$upper `:` type($upper)`)` (`(` $lb^ `:` $ub`)`)? attr-dict-with-keyword `:` type($res)
  }];

  let verifier = [{
    return ::verify(*this);
  }];

  let extraClassDeclaration = [{
    Operation* getLowerOp() { return lower().front().getDefiningOp(); }
    Operation* getUpperOp() { return upper().front().getDefiningOp(); }
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    stencil::CombineOp getCombineTreeRoot();
  }];
}

#endif // STENCIL_OPS
