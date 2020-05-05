#include "Dialect/Stencil/StencilUtils.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"

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
  assert((value.getType().isa<stencil::FieldType>() ||
          value.getType().isa<stencil::TempType>()) &&
         "expected stencil field or temp type");

  if (value.getType().isa<stencil::FieldType>())
    return value.getType().cast<stencil::FieldType>().getElementType();
  if (value.getType().isa<stencil::TempType>())
    return value.getType().cast<stencil::TempType>().getElementType();
  return {};
}

ArrayRef<int> getDimensions(Value value) {
  assert((value.getType().isa<stencil::FieldType>() ||
          value.getType().isa<stencil::TempType>()) &&
         "expected stencil field or temp type");

  if (value.getType().isa<stencil::FieldType>())
    return value.getType().cast<stencil::FieldType>().getShape();
  if (value.getType().isa<stencil::TempType>())
    return value.getType().cast<stencil::TempType>().getShape();
  return {};
}

} // namespace stencil
} // namespace mlir
