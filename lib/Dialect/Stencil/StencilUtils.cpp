#include "Dialect/Stencil/StencilUtils.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"
#include <bits/stdint-intn.h>

namespace mlir {
namespace stencil {

ArrayAttr convertVecToAttr(ArrayRef<int64_t> vector, MLIRContext *context) {
  SmallVector<Attribute, 3> result;
  for (int64_t value : vector) {
    result.push_back(IntegerAttr::get(IntegerType::get(64, context), value));
  }
  return ArrayAttr::get(result, context);
}
SmallVector<int64_t, 3> convertAttrToVec(ArrayAttr attr) {
  SmallVector<int64_t, 3> result;
  for (auto &elem : attr) {
    result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
  }
  return result;
}

SmallVector<int64_t, 3> convertAttrToVec(Optional<ArrayAttr> attr) {
  assert(attr.hasValue() && "expected optional attribute to have value");
  return convertAttrToVec(attr.getValue());
}

Type getElementType(Value value) {
  assert(value.getType().isa<stencil::GridType>() &&
         "expected stencil field or temp type");
  return value.getType().cast<stencil::GridType>().getElementType();
}

ArrayRef<int64_t> getShape(Value value) {
  assert(value.getType().isa<stencil::GridType>() &&
         "expected stencil field or temp type");
  return value.getType().cast<stencil::GridType>().getShape();
}

} // namespace stencil
} // namespace mlir
