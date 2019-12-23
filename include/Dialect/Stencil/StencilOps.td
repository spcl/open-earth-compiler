#ifndef STENCIL_OPS
#define STENCIL_OPS

#ifndef STENCIL_BASE
include "Dialect/Stencil/StencilBase.td"
#endif // STENCIL_BASE

def Stencil_AccessOp : Stencil_Op<"access", [NoSideEffect]> {
  let summary = "view access operation";
  let description = [{
    This operation takes a view as an input as well as an offset
    attribute and return the corresponding element from the view's
    underlying field. The offset is specified relatively to the
    current position.

    Example:

      %0 = stencil.access %view[-1, 0, 0] : !stencil.view<ijk,f64>
  }];

  let arguments = (ins Stencil_View:$view, I64ArrayAttr:$offset);
  let results = (outs Stencil_ElementType:$res);

  let builders = [OpBuilder<"Builder *, OperationState &, Value *,"
                            "ArrayRef<int64_t>">];

  let extraClassDeclaration = [{
    static StringRef getOffsetAttrName() { return "offset"; }
    SmallVector<int64_t, 3> getOffset() {
      return convertArrayAttrToSmallVector(offset().cast<ArrayAttr>(), getContext());
    }
  }];
}

def Stencil_LoadOp : Stencil_Op<"load", [NoSideEffect]> {
  let summary = "load operation";
  let description = [{
    This operation takes a field and returns a view.

    Example:

      %0 = stencil.load %field : !stencil.view<ijk,f64>
  }];

  let arguments = (ins Stencil_Field:$field,
                        OptionalAttr<I64ArrayAttr>:$lb, 
                        OptionalAttr<I64ArrayAttr>:$ub);
  let results = (outs Stencil_View:$res);

  let builders = [OpBuilder<"Builder *, OperationState &, Value *">];

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
    void setLB(SmallVector<int64_t, 3> bound) {
      setAttr(stencil::LoadOp::getLBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
    }
    void setUB(SmallVector<int64_t, 3> bound) {
      setAttr(stencil::LoadOp::getUBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
    }
    stencil::ViewType getResultViewType() {
      return res()->getType().cast<stencil::ViewType>();
    }
  }];
}

def Stencil_StoreOp : Stencil_Op<"store"> {
  let summary = "store operation";
  let description = [{
    This operation takes a view and writes a field on a user defined range.

    Example:

      %0 = stencil.store %view to %field[0,0,0][64,64,60] : !stencil.field<ijk,f64>
  }];

  let arguments = (ins Stencil_View:$view, Stencil_Field:$field, 
                        I64ArrayAttr:$lb, I64ArrayAttr:$ub);
  let results = (outs);

  let builders = [OpBuilder<"Builder *, OperationState &, Value *,"
                            "Value *, ArrayRef<int64_t>, ArrayRef<int64_t>">];

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    SmallVector<int64_t, 3> getLB() {
      return convertArrayAttrToSmallVector(lb().cast<ArrayAttr>(), getContext());
    }
    SmallVector<int64_t, 3> getUB() {
      return convertArrayAttrToSmallVector(ub().cast<ArrayAttr>(), getContext());
    }
    stencil::FieldType getFieldType() {
      return field()->getType().cast<stencil::FieldType>();
    }
    stencil::ViewType getViewType() {
      return view()->getType().cast<stencil::ViewType>();
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
    the stencil function to the output view.

    Example:

      %0 = stencil.appy %arg0=%0 : !stencil.view<ijk,f64> {
        ...
      } : !stencil.view<ijk,f64>
  }];

  let arguments = (ins Variadic<AnyType>:$operands, 
                        OptionalAttr<I64ArrayAttr>:$lb, 
                        OptionalAttr<I64ArrayAttr>:$ub);
  let results = (outs Variadic<Stencil_View>:$res);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result,"
              "Block *body, ArrayRef<Value *> operands = {}">];

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
    void setLB(SmallVector<int64_t, 3> bound) {
      setAttr(stencil::ApplyOp::getLBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
    }
    void setUB(SmallVector<int64_t, 3> bound) {
      setAttr(stencil::ApplyOp::getUBAttrName(), convertSmallVectorToArrayAttr(bound, getContext()));
    }
    stencil::ViewType getResultViewType(unsigned i) {
      return res()[i]->getType().cast<stencil::ViewType>();
    }
    /// Get the body of the apply op
    Block *getBody() { return &region().front(); }
  }];
}

def Stencil_CallOp : Stencil_Op<"call", [NoSideEffect]> {
  let summary = "call operation";
  let description = [{
    This operation takes a stencil function plus parameters and evaluates 
    the function at the given offset.

    Example:

      %0 = stencil.call @func(%0) : (!stencil.view<ijk,f64>) -> !stencil.view<ijk,f64>
  }];

  let arguments = (ins SymbolRefAttr:$callee, 
                        I64ArrayAttr:$offset, Variadic<AnyType>:$operands);
  let results = (outs Stencil_ElementType:$res);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, FuncOp callee,"
              "stencil::ViewType viewType, ArrayRef<int64_t> offset,"
              "ArrayRef<Value *> operands = {}">];

  let extraClassDeclaration = [{
    stencil::ViewType getResultViewType() {
      return res()->getType().cast<stencil::ViewType>();
    }

    static StringRef getCalleeAttrName() { return "callee"; }
    FunctionType getCalleeType();
    FuncOp getCallee() {
      return getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(callee().getLeafReference());
    }
    static StringRef getOffsetAttrName() { return "offset"; }
    SmallVector<int64_t, 3> getOffset() {
      return convertArrayAttrToSmallVector(offset().cast<ArrayAttr>(), getContext());
    }

    /// Get the argument operands to the called function.
    // operand_range getArgOperands() {
    //   return {arg_operand_begin(), arg_operand_end()};
    // }
    // operand_iterator arg_operand_begin() { return operand_begin(); }
    // operand_iterator arg_operand_end() { return operand_end(); }
  }];
}

def Stencil_ReturnOp : Stencil_Op<"return", [Terminator, HasParent<"ApplyOp">]> {
  let summary = "return operation";
  let description = [{
    This operation represents the return operation of a stencil apply.
    The operation takes a variable number of operands that must match
    the element types of the returned views.

      stencil.return %0 : f64
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
}

#endif // STENCIL_OPS
