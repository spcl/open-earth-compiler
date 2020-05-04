#include "Dialect/Stencil/StencilUtils.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace stencil {

// Convert a vector to an attribute
ArrayAttr convertVecToAttr(ArrayRef<int64_t> vector, MLIRContext *context) {
  SmallVector<Attribute, 3> result;
  for (int64_t value : vector) {
    result.push_back(IntegerAttr::get(IntegerType::get(64, context), value));
  }
  return ArrayAttr::get(result, context);
}

// Convert an attribute to a vector
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

} // namespace stencil
} // namespace mlir
