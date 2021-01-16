#include "Dialect/Stencil/StencilUtils.h"
#include "mlir/Support/LLVM.h"
#include <cstdint>
#include <functional>

namespace mlir {
namespace stencil {

int64_t min(int64_t x, int64_t y) { return std::min(x, y); }
int64_t max(int64_t x, int64_t y) { return std::max(x, y); }

Index applyFunElementWise(ArrayRef<int64_t> x, ArrayRef<int64_t> y,
                          std::function<int64_t(int64_t, int64_t)> fun) {
  assert(x.size() == y.size() && "expected the indexes to have the same size");
  Index result(x.size());
  llvm::transform(llvm::zip(x, y), result.begin(),
                  [&](std::tuple<int64_t, int64_t> x) {
                    return fun(std::get<0>(x), std::get<1>(x));
                  });
  return result;
}

} // namespace stencil
} // namespace mlir
