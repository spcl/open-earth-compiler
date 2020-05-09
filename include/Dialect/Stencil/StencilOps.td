#ifndef STENCIL_OPS
#define STENCIL_OPS

include "Dialect/Stencil/StencilBase.td"
include "Dialect/Stencil/StencilInterfaces.td"
include "mlir/Interfaces/SideEffects.td"

//===----------------------------------------------------------------------===//
// Concrete Operations
//===----------------------------------------------------------------------===//

def Stencil_AssertOp : Stencil_Op<"assert", [DeclareOpInterfaceMethods<ShapeAccess>]> {
  let summary = "assert the input field size";
  let description = [{
    This operation asserts the size of input fields.

    Example:
      stencil.assert %in ([-3, -3, 0] : [67, 67, 60]) : !stencil.field<?x?x?xf64>
  }];

  let arguments = (ins Stencil_Field:$field, Stencil_Index:$lb, Stencil_Index:$ub);
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "Value field, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub", [{
      state.addOperands(field);
      state.addAttribute(getLBAttrName(), builder.getI64ArrayAttr(lb));
      state.addAttribute(getUBAttrName(), builder.getI64ArrayAttr(ub));
    }]>
  ];

  let assemblyFormat = [{
    $field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` type($field)
  }];

  let verifier = [{
    auto fieldType = field().getType().cast<stencil::GridType>();
    auto shapeOp = cast<ShapeAccess>(this->getOperation());
    if(fieldType.hasStaticShape())
      return emitOpError("expected fields to have dynamic shape");
    if(shapeOp.getRank() != fieldType.getRank())
      return emitOpError("expected op and field to have the same rank");
    
    // Verify all users fit the shape
    for(auto user : field().getUsers()) {
      if(auto userOp = dyn_cast<ShapeAccess>(user)) {
        if(userOp.hasShape() &&
           (shapeOp.getLB() != mapFunctionToIdxPair(shapeOp.getLB(), userOp.getLB(), minimum) ||
            shapeOp.getUB() != mapFunctionToIdxPair(shapeOp.getUB(), userOp.getUB(), maximum))) 
          return emitOpError("asserted shape not large enough to fit all accesses");
      }
    }

    if (llvm::count_if(field().getUsers(), hasOpType<stencil::AssertOp>) != 1)
      return emitOpError("expected exactly one assert operation");
    return success();
  }];
  
  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    Index getLB() {
      return convertAttrToIndex(lb()); 
    }
    Index getUB() {
      return convertAttrToIndex(ub());
    }
    void setLB(ArrayRef<int64_t> bound) {
      lbAttr(convertIndexToAttr(bound, getContext()));
    }
    void setUB(ArrayRef<int64_t> bound) {
      ubAttr(convertIndexToAttr(bound, getContext()));
    }
    stencil::FieldType getFieldType() {
      return field().getType().cast<stencil::FieldType>();
    }
  }];
}

def Stencil_AccessOp : Stencil_Op<"access", [NoSideEffect]> {
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
      state.addTypes(getElementType(temp));
    }]>
  ];

  let assemblyFormat = [{
    $temp $offset attr-dict-with-keyword `:` functional-type($temp, $res)
  }];

  let verifier = [{
    if (offset().size() != getShape(temp()).size())
      return emitOpError("offset and temp dimensions do not match");
    if (res().getType() != getElementType(temp()))
      return emitOpError("result type and element type are inconsistent");
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getOffsetAttrName() { return "offset"; }
    Index getOffset() {
      return convertAttrToIndex(offset());
    }
    void setOffset(ArrayRef<int64_t> bound) {
      setAttr(stencil::AccessOp::getOffsetAttrName(), convertIndexToAttr(bound, getContext()));
    }
  }];
}

def Stencil_LoadOp : Stencil_Op<"load", [DeclareOpInterfaceMethods<ShapeAccess>,
                                         DeclareOpInterfaceMethods<ShapeInference>,
                                         NoSideEffect]> {
  let summary = "load operation";
  let description = [{
    This operation takes a field and returns a temporary values.

    Example:
      %0 = stencil.load %field : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  }];

  let arguments = (ins Stencil_Field:$field,
                       OptionalAttr<Stencil_Index>:$lb, 
                       OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Stencil_Temp:$res);
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Value field", [{
      state.addOperands(field);
      state.addTypes(stencil::TempType::get(builder.getContext(), 
                     getElementType(field), getShape(field)));
    }]>
  ];

  let assemblyFormat = [{
    $field (`(` $lb^ `:` $ub `)`)? attr-dict-with-keyword `:` functional-type($field, $res)
  }];

  let verifier = [{
    // Check the field and result types
    auto fieldType = field().getType().cast<stencil::GridType>();
    auto resType = res().getType().cast<stencil::GridType>();
    if (fieldType.getRank() != resType.getRank())
      return emitOpError("the field and temp types have different rank");
    if (fieldType.getAllocation() != resType.getAllocation())
      return emitOpError("the field and temp types have different allocation");
    if (fieldType.getElementType() != resType.getElementType())
      return emitOpError("the field and temp types have different element type");
    
    if (llvm::count_if(field().getUsers(), hasOpType<stencil::AssertOp>) != 1)
      return emitOpError("expected exactly one assert operation");
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    Index getLB() {
      return convertAttrToIndex(lb());
    }
    Index getUB() {
      return convertAttrToIndex(ub());
    }
    void setLB(ArrayRef<int64_t> bound) {
      lbAttr(convertIndexToAttr(bound, getContext()));
    }
    void setUB(ArrayRef<int64_t> bound) {
      ubAttr(convertIndexToAttr(bound, getContext()));
    }
  }];
}

def Stencil_StoreOp : Stencil_Op<"store", [DeclareOpInterfaceMethods<ShapeAccess>]> {
  let summary = "store operation";
  let description = [{
    This operation takes a temp and writes a field on a user defined range.

    Example:
      stencil.store %temp to %field ([0,0,0] : [64,64,60]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
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
    if (fieldType.getRank() != tempType.getRank())
      return emitOpError("the field and temp types have different rank");
    if (fieldType.getAllocation() != tempType.getAllocation())
      return emitOpError("the field and temp types have different allocation");
    if (fieldType.getElementType() != tempType.getElementType())
      return emitOpError("the field and temp types have different element type");
    
    if (!dyn_cast<stencil::ApplyOp>(temp().getDefiningOp()))
      return emitOpError("output temp not result of an apply");
    if (llvm::count_if(field().getUsers(), hasOpType<stencil::AssertOp>) != 1)
      return emitOpError("expected exactly one assert operation");
    if (llvm::count_if(field().getUsers(), hasOpType<stencil::LoadOp>) != 0)
      return emitOpError("an output cannot be an input");  
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    Index getLB() {
      return convertAttrToIndex(lb());
    }
    Index getUB() {
      return convertAttrToIndex(ub());
    }
    void setLB(ArrayRef<int64_t> bound) {
      lbAttr(convertIndexToAttr(bound, getContext()));
    }
    void setUB(ArrayRef<int64_t> bound) {
      ubAttr(convertIndexToAttr(bound, getContext()));
    }
  }];
}

def Stencil_ApplyOp : Stencil_Op<"apply", [DeclareOpInterfaceMethods<ShapeAccess>,
                                           DeclareOpInterfaceMethods<ShapeInference>,
                                           IsolatedFromAbove, 
                                           SingleBlockImplicitTerminator<"ReturnOp">,
                                           NoSideEffect]> {
  let summary = "apply operation";
  let description = [{
    This operation takes a stencil function plus parameters and applies 
    the stencil function to the output temp.

    Example:

      %0 = stencil.apply %arg0=%0 : !stencil.temp<?x?x?xf64> {
        ...
      } : !stencil.temp<?x?x?xf64>
  }];

  let arguments = (ins Variadic<AnyType>:$operands, 
                        OptionalAttr<Stencil_Index>:$lb, 
                        OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Variadic<Stencil_Temp>:$res);
  let regions = (region SizedRegion<1>:$region);
  let hasCanonicalizer = 1;
  
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state,"
              "ValueRange operands, ValueRange results", [{
      state.addOperands(operands);
      auto region = state.addRegion(); 
      region->push_back(new Block());
      for(auto operand : operands) {
        region->front().addArgument(operand.getType());
      }

      SmallVector<Type, 3> resultTypes;
      resultTypes.resize(results.size());
      llvm::transform(results, resultTypes.begin(),
                      [](Value value) { return value.getType(); });
      state.addTypes(resultTypes);
    }]>
  ];

  let verifier = [{
    // TODO check all results have the same dimension
    // TODO check this dimension matches the range dim
    
    if (region().front().getNumArguments() != operands().size())
      return emitOpError("operand and argument counts do not match");
      
    for (unsigned i = 0, e = operands().size(); i != e; ++i) {
      if (region().front().getArgument(i).getType() !=
          operands()[i].getType())
        return emitOpError("operand and argument types do not match");
    }
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    Index getLB() {
      return convertAttrToIndex(lb());
    }
    Index getUB() {
      return convertAttrToIndex(ub());
    }
    void setLB(ArrayRef<int64_t> bound) {
      lbAttr(convertIndexToAttr(bound, getContext()));
    }
    void setUB(ArrayRef<int64_t> bound) {
      ubAttr(convertIndexToAttr(bound, getContext()));
    }
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
  
    // Verify the number of operands matches the number of apply results
    auto results = applyOp.res();
    if (getNumOperands() != unrollFactor * results.size())
      return emitOpError("the operand and apply result counts do not match");
  
    // Verify the element types match
    for (unsigned i = 0, e = results.size(); i != e; ++i) {
      for (unsigned j = 0; j < unrollFactor; j++)
        if (getOperand(i * unrollFactor + j).getType() != getElementType(applyOp.getResult(i)))
          return emitOpError("the operand and apply result element types do not match");
    }
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getUnrollAttrName() { return "unroll"; }
    Index getUnroll() {
      return convertAttrToIndex(unroll());
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
  }];
}

#endif // STENCIL_OPS
