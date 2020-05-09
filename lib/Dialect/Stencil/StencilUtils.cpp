#include "Dialect/Stencil/StencilUtils.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"
#include <bits/stdint-intn.h>
#include <functional>

namespace mlir {
namespace stencil {

int64_t minimum(int64_t x, int64_t y) { return std::min(x, y); }
int64_t maximum(int64_t x, int64_t y) { return std::max(x, y); }

Index mapFunctionToIdxPair(ArrayRef<int64_t> x, ArrayRef<int64_t> y,
                           std::function<int64_t(int64_t, int64_t)> fun) {
  assert(x.size() == y.size() && "expected the indexes to have the same size");
  Index result(x.size());
  llvm::transform(llvm::zip(x, y), result.begin(),
                  [&](std::tuple<int64_t, int64_t> x) {
                    return fun(std::get<0>(x), std::get<1>(x));
                  });
  return result;
}

// TODO remove below
ArrayAttr convertIndexToAttr(ArrayRef<int64_t> index, MLIRContext *context) {
  SmallVector<Attribute, 3> result;
  for (int64_t value : index) {
    result.push_back(IntegerAttr::get(IntegerType::get(64, context), value));
  }
  return ArrayAttr::get(result, context);
}
Index convertAttrToIndex(ArrayAttr attr) {
  Index result;
  for (auto &elem : attr) {
    result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
  }
  return result;
}
Index convertAttrToIndex(Optional<ArrayAttr> attr) {
  assert(attr.hasValue() && "expected optional attribute to have value");
  return convertAttrToIndex(attr.getValue());
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
