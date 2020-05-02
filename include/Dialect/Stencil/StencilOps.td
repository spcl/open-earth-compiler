#ifndef STENCIL_OPS
#define STENCIL_OPS

#ifndef STENCIL_BASE
include "Dialect/Stencil/StencilBase.td"
#endif // STENCIL_BASE

def Stencil_AssertOp : Stencil_Op<"assert", []> {
  let summary = "temp access operation";
  let description = [{
    This operation asserts the size of input fields.

    Example:

      stencil.assert %in ([-3, -3, 0]:[67, 67, 60]) : !stencil.field<ijk,f64>
  }];

  let arguments = (ins Stencil_Field:$field, I64ArrayAttr:$lb, I64ArrayAttr:$ub);

  let builders = [OpBuilder<"OpBuilder &, OperationState &, Value,"
                            "ArrayRef<int64_t>, ArrayRef<int64_t>">];

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
      setAttr(stencil::AssertOp::getLBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
    }
    void setUB(ArrayRef<int64_t> bound) {
      setAttr(stencil::AssertOp::getUBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
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
      %0 = stencil.access %temp[-1, 0, 0] : !stencil.temp<ijk,f64> -> f64
  }];

  let arguments = (
    ins Stencil_Temp:$temp, 
    Confined<I64ArrayAttr, [ArrayCount<3>]>:$offset);

  let results = (
    outs Stencil_ElementType:$res);
  
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
    Type elementType = this->temp().getType().cast<stencil::TempType>().getElementType();
    Type resultType = this->res().getType();
    if (resultType != elementType)
      return this->emitOpError("result type ")
             << resultType << " and element type " << elementType << " are inconsistent";
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getOffsetAttrName() { 
      return "offset"; 
    }
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
    This operation takes a field and returns a temp.

    Example:

      %0 = stencil.load %field : (!stencil.field<ijk,f64>) -> !stencil.temp<ijk,f64>
  }];

  let arguments = (ins Stencil_Field:$field,
                        OptionalAttr<I64ArrayAttr>:$lb, 
                        OptionalAttr<I64ArrayAttr>:$ub);
  let results = (outs Stencil_Temp:$res);

  let builders = [OpBuilder<"OpBuilder &, OperationState &, Value">];

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

  let arguments = (ins Stencil_Temp:$temp, Stencil_Field:$field, 
                        I64ArrayAttr:$lb, I64ArrayAttr:$ub);
  let results = (outs);

  let builders = [OpBuilder<"OpBuilder &, OperationState &, Value,"
                            "Value, ArrayRef<int64_t>, ArrayRef<int64_t>">];

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

def Stencil_ApplyOp : Stencil_Op<"apply", [
                        IsolatedFromAbove, 
                        NoSideEffect, 
                        SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = "apply operation";
  let description = [{
    This operation takes a stencil function plus parameters and applies 
    the stencil function to the output temp.

    Example:

      %0 = stencil.appy %arg0=%0 : !stencil.temp<ijk,f64> {
        ...
      } : !stencil.temp<ijk,f64>
  }];

  let arguments = (ins Variadic<AnyType>:$operands, 
                        OptionalAttr<I64ArrayAttr>:$lb, 
                        OptionalAttr<I64ArrayAttr>:$ub);
  let results = (outs Variadic<Stencil_Temp>:$res);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result,"
              "ValueRange operands, ValueRange results">];

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
    This operation represents the return operation of a stencil apply.
    The operation takes a variable number of operands that must match
    the element types of the returned temps.

    The optional unroll attribute should never be used when writing stencils by
    hand. Instead it is used during the automatic optimization.

    Examples:

      stencil.return %0 : f64
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                   OptionalAttr<I64ArrayAttr>:$unroll);

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
