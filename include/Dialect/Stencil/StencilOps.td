#ifndef STENCIL_OPS
#define STENCIL_OPS

include "Dialect/Stencil/StencilBase.td"
include "mlir/Interfaces/SideEffects.td"

//===----------------------------------------------------------------------===//
// Concrete Operations
//===----------------------------------------------------------------------===//

def Stencil_AssertOp : Stencil_Op<"assert", []> {
  let summary = "assert the input field size";
  let description = [{
    This operation asserts the size of input fields.

    Example:
      stencil.assert %in [-3, -3, 0] : [67, 67, 60] : !stencil.field<ijk,f64>
  }];

  let arguments = (ins Stencil_Field:$field, Stencil_Index:$lb, Stencil_Index:$ub);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "Value field, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub", [{
      state.addOperands(field);
      state.addAttribute(getLBAttrName(), builder.getI64ArrayAttr(lb));
      state.addAttribute(getUBAttrName(), builder.getI64ArrayAttr(ub));
    }]>
  ];

  let assemblyFormat = [{
    $field `(` $lb `:` $ub `)` attr-dict `:` type($field)
  }];

  let verifier = [{
    for (auto user : field().getUsers()) {
      if (isa<stencil::AssertOp>(user) && user != getOperation())
        return emitOpError("multiple asserts for the same field");
    }
    return success();
  }];
  
  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    SmallVector<int64_t, 3> getLB() {
      return convertArrayAttrToSmallVector(lb().cast<ArrayAttr>(), getContext());
    }
    SmallVector<int64_t, 3> getUB() {
      return convertArrayAttrToSmallVector(ub().cast<ArrayAttr>(), getContext());
    }
    void setLB(ArrayRef<int64_t> bound) {
      setAttr(stencil::AssertOp::getLBAttrName(), 
              convertSmallVectorToArrayAttr(bound, getContext()));
    }
    void setUB(ArrayRef<int64_t> bound) {
      setAttr(stencil::AssertOp::getUBAttrName(), 
              convertSmallVectorToArrayAttr(bound, getContext()));
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
      %0 = stencil.access %temp [-1, 0, 0] : !stencil.temp<ijk,f64> -> f64
  }];

  let arguments = (ins Stencil_Temp:$temp, Stencil_Index:$offset);
  let results = (outs Stencil_ElementType:$res);
  
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "Value temp, ArrayRef<int64_t> offset", [{
      state.addOperands(temp);
      state.addAttribute(getOffsetAttrName(), builder.getI64ArrayAttr(offset));
      state.addTypes(temp.getType().cast<stencil::TempType>().getElementType());
    }]>
  ];

  let assemblyFormat = [{
    $temp $offset attr-dict `:` functional-type($temp, $res)
  }];

  let verifier = [{
    Type elementType = getTempType().getElementType();
    Type resultType = res().getType();
    if (resultType != elementType)
      return emitOpError("result type and element type are inconsistent");
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getOffsetAttrName() { return "offset"; }
    SmallVector<int64_t, 3> getOffset() {
      return convertArrayAttrToSmallVector(offset().cast<ArrayAttr>(), 
                                           getContext());
    }
    void setOffset(ArrayRef<int64_t> bound) {
      setAttr(stencil::AccessOp::getOffsetAttrName(), 
              convertSmallVectorToArrayAttr(bound, getContext()));
    }
    stencil::TempType getTempType() {
      return temp().getType().cast<stencil::TempType>();
    }
  }];
}

def Stencil_LoadOp : Stencil_Op<"load", [NoSideEffect]> {
  let summary = "load operation";
  let description = [{
    This operation takes a field and returns a temporary values.

    Example:
      %0 = stencil.load %field : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  }];

  let arguments = (ins Stencil_Field:$field,
                       OptionalAttr<Stencil_Index>:$lb, 
                       OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Stencil_Temp:$res);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Value field", [{
      state.addOperands(field);
      stencil::FieldType fieldType = field.getType().cast<stencil::FieldType>();
      state.addTypes(stencil::TempType::get(builder.getContext(), 
                     fieldType.getElementType(), fieldType.getDimensions()));
    }]>
  ];

  let assemblyFormat = [{
    $field (`(` $lb^ `:` $ub `)`)? attr-dict `:` functional-type($field, $res)
  }];

  let verifier = [{
    stencil::FieldType fieldType = field().getType().cast<stencil::FieldType>();
    stencil::TempType tempType = res().getType().cast<stencil::TempType>();
    if (fieldType.getElementType() != tempType.getElementType())
      return emitOpError("the field and temp element types do not match");
    if (fieldType.getDimensions() != tempType.getDimensions())
      return emitOpError("the field and temp dimensions do not match");
  
    int asserts = 0;
    for (OpOperand &use : field().getUses()) {
      if (auto assertOp = dyn_cast<stencil::AssertOp>(use.getOwner()))
        asserts++;
    }
    if (asserts != 1)
      return emitOpError("expected exactly one assert operation");
  
    return success();
  }];


  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    SmallVector<int64_t, 3> getLB() {
      assert(lb().hasValue() && "expected lb to have value");
      return convertArrayAttrToSmallVector(lb().getValue(), getContext());
    }
    SmallVector<int64_t, 3> getUB() {
      assert(ub().hasValue() && "expected ub to have value");
      return convertArrayAttrToSmallVector(ub().getValue(), getContext());
    }
    void setLB(ArrayRef<int64_t> bound) {
      setAttr(stencil::LoadOp::getLBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
    }
    void setUB(ArrayRef<int64_t> bound) {
      setAttr(stencil::LoadOp::getUBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
    }
    stencil::TempType getResultTempType() {
      return res().getType().cast<stencil::TempType>();
    }
  }];
}

def Stencil_StoreOp : Stencil_Op<"store"> {
  let summary = "store operation";
  let description = [{
    This operation takes a temp and writes a field on a user defined range.

    Example:
      stencil.store %temp to %field[0,0,0][64,64,60] : !stencil.temp<ijk,f64> to !stencil.field<ijk,f64>
  }];

  let arguments = (ins Stencil_Temp:$temp, 
                       Stencil_Field:$field, 
                       Stencil_Index:$lb, 
                       Stencil_Index:$ub);
  let results = (outs);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Value temp, "
              "Value field, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub", [{
      state.addOperands({temp, field});
      state.addAttribute(getLBAttrName(), builder.getI64ArrayAttr(lb));
      state.addAttribute(getUBAttrName(), builder.getI64ArrayAttr(ub)); 
    }]>
  ];
  
  let assemblyFormat = [{
    $temp `to` $field `(` $lb `:` $ub `)` attr-dict `:` type($temp) `to` type($field)
  }];

  let verifier = [{
    stencil::FieldType fieldType = field().getType().cast<stencil::FieldType>();
    stencil::TempType tempType = temp().getType().cast<stencil::TempType>();
    if (fieldType.getElementType() != tempType.getElementType())
      return emitOpError("the field and temp element types do not match");
    if (fieldType.getDimensions() != tempType.getDimensions())
      return emitOpError("the field and temp dimensions do not match");
  
    if (!dyn_cast<stencil::ApplyOp>(temp().getDefiningOp()))
      return emitOpError("output temp not result of an apply");

    // TODO check no load and a single store!
    int asserts = 0;
    for (OpOperand &use : field().getUses()) {
      if (auto assertOp = dyn_cast<stencil::AssertOp>(use.getOwner()))
        asserts++;
    }
    if (asserts != 1)
      return emitOpError("expected exactly one assert operation");
  
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    SmallVector<int64_t, 3> getLB() {
      return convertArrayAttrToSmallVector(lb().cast<ArrayAttr>(), getContext());
    }
    SmallVector<int64_t, 3> getUB() {
      return convertArrayAttrToSmallVector(ub().cast<ArrayAttr>(), getContext());
    }
    void setLB(ArrayRef<int64_t> bound) {
      setAttr(stencil::StoreOp::getLBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
    }
    void setUB(ArrayRef<int64_t> bound) {
      setAttr(stencil::StoreOp::getUBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
    }
    stencil::FieldType getFieldType() {
      return field().getType().cast<stencil::FieldType>();
    }
    stencil::TempType getTempType() {
      return temp().getType().cast<stencil::TempType>();
    }
  }];
}

def Stencil_ApplyOp : Stencil_Op<"apply", [IsolatedFromAbove, 
                                           SingleBlockImplicitTerminator<"ReturnOp">,
                                           NoSideEffect]> {
  let summary = "apply operation";
  let description = [{
    This operation takes a stencil function plus parameters and applies 
    the stencil function to the output temp.

    Example:

      %0 = stencil.apply %arg0=%0 : !stencil.temp<ijk,f64> {
        ...
      } : !stencil.temp<ijk,f64>
  }];

  let arguments = (ins Variadic<AnyType>:$operands, 
                        OptionalAttr<Stencil_Index>:$lb, 
                        OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Variadic<Stencil_Temp>:$res);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state,"
              "ValueRange operands, ValueRange results", [{
      state.addOperands(operands);
      Region *body = state.addRegion(); // Add an empty body
  
      SmallVector<Type, 3> resultTypes;
      resultTypes.resize(results.size());
      llvm::transform(results, std::begin(resultTypes),
                      [](Value value) { return value.getType(); });
      state.addTypes(resultTypes);
    }]>
  ];

  let verifier = [{
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
    static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    SmallVector<int64_t, 3> getLB() {
      assert(lb().hasValue() && "expected lb to have value");
      return convertArrayAttrToSmallVector(lb().getValue(), getContext());
    }
    SmallVector<int64_t, 3> getUB() {
      assert(ub().hasValue() && "expected ub to have value");
      return convertArrayAttrToSmallVector(ub().getValue(), getContext());
    }
    void setLB(ArrayRef<int64_t> bound) {
      setAttr(stencil::ApplyOp::getLBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
    }
    void setUB(ArrayRef<int64_t> bound) {
      setAttr(stencil::ApplyOp::getUBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
    }
    stencil::TempType getResultTempType(unsigned i) {
      return res()[i].getType().cast<stencil::TempType>();
    }
    /// Get the body of the apply op
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

  let arguments = (ins Variadic<AnyType>:$operands,
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
    (`unroll` $unroll^)? $operands attr-dict `:` type($operands)
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
        if (getOperand(i * unrollFactor + j).getType() !=
            applyOp.getResultTempType(i).getElementType())
          return emitOpError("the operand and apply result element types do not match");
    }
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getUnrollAttrName() { return "unroll"; }
    SmallVector<int64_t, 3> getUnroll() {
      assert(unroll().hasValue() && "expected unroll to have value");
      return convertArrayAttrToSmallVector(unroll().getValue(), getContext());
    }
    unsigned getUnrollFactor() {
      unsigned factor = 1;
      if (unroll().hasValue()) {
        SmallVector<int64_t, 3> unroll = getUnroll();
        factor = std::accumulate(unroll.begin(), unroll.end(), 1,
                                      std::multiplies<int64_t>());
      }
      return factor;
    }
  }];
}

#endif // STENCIL_OPS
