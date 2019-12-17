#ifndef STENCIL_OPS
#define STENCIL_OPS

#ifndef STENCIL_BASE
include "Dialect/Stencil/StencilBase.td"
#endif // STENCIL_BASE

def Stencil_AccessOp : Stencil_Op<"access"> {
  let summary = "view access operation";
  let description = [{
    This operation takes a view as an input as well as an offset
    attribute and return the corresponding element from the view's
    underlying field. The offset is specified relatively to the
    current position.

    Example:

      %0 = stencil.access %view[-1, 0, 0] : !stencil.view<?x?x?xf64>
  }];

  let arguments = (ins Stencil_View:$view, I64ArrayAttr:$offset);
  let results = (outs Stencil_ElementType:$res);

  let builders = [OpBuilder<"Builder *, OperationState &, Value *,"
                            "ArrayRef<int64_t>">];

  let extraClassDeclaration = [{
    static StringRef getOffsetAttrName() { return "offset"; }
    SmallVector<int64_t, 3> getOffset() {
      SmallVector<int64_t, 3> offsetAttr;
      for(auto &attr : offset().cast<ArrayAttr>().getValue()) {
        offsetAttr.push_back(attr.cast<IntegerAttr>().getValue().getSExtValue());
      }
      return offsetAttr;
    }
  }];
}

def Stencil_LoadOp : Stencil_Op<"load"> {
  let summary = "load operation";
  let description = [{
    This operation takes a field and returns a view.

    Example:

      %0 = stencil.load %field : (!stencil.field<?x?x?xf64>) -> !stencil.view<?x?x?xf64>
  }];

  let arguments = (ins Stencil_Field:$field);
  let results = (outs Stencil_View:$res);

  let builders = [OpBuilder<"Builder *, OperationState &, Value *,"
                            "ArrayRef<int64_t> = {-1, -1, -1}">];

  let extraClassDeclaration = [{
    stencil::ViewType getResultViewType() {
      return res()->getType().cast<stencil::ViewType>();
    }
  }];
}

def Stencil_StoreOp : Stencil_Op<"store"> {
  let summary = "store operation";
  let description = [{
    This operation takes a field and returns a view.

    Example:

      %0 = stencil.store %view to %field : !stencil.view<?x?x?xf64>, !stencil.field<12x12x16xf64>
  }];

  let arguments = (ins Stencil_View:$view, Stencil_Field:$field);
  let results = (outs);

  let extraClassDeclaration = [{
    stencil::FieldType getFieldType() {
      return field()->getType().cast<stencil::FieldType>();
    }
    stencil::ViewType getViewType() {
      return view()->getType().cast<stencil::ViewType>();
    }
  }];
}

def Stencil_ApplyOp : Stencil_Op<"apply"> {
  let summary = "apply operation";
  let description = [{
    This operation takes a stencil function plus parameters and applies 
    the stencil function to the output view.

    Example:

      %0 = stencil.appy @func(%0) : (!stencil.view<?x?x?xf64>) -> !stencil.view<?x?x?xf64>
  }];

  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Stencil_View:$res);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, FuncOp callee,"
              "stencil::ViewType viewType, ArrayRef<Value *> operands = {}">];

  let extraClassDeclaration = [{
    stencil::ViewType getResultViewType() {
      return res()->getType().cast<stencil::ViewType>();
    }

    static StringRef getCalleeAttrName() { return "callee"; }
    FunctionType getCalleeType();
    FuncOp getCallee() {
        return getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(callee().getLeafReference());
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];
}

def Stencil_CallOp : Stencil_Op<"call"> {
  let arguments = (ins SymbolRefAttr:$callee, I64ArrayAttr:$offset, Variadic<AnyType>:$operands);
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
      SmallVector<int64_t, 3> offsetAttr;
      for(auto &attr : offset().cast<ArrayAttr>().getValue()) {
        offsetAttr.push_back(attr.cast<IntegerAttr>().getValue().getSExtValue());
      }
      return offsetAttr;
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];
}

#endif // STENCIL_OPS
