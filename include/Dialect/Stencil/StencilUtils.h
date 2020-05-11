#ifndef DIALECT_STENCIL_STENCILUTILS_H
#define DIALECT_STENCIL_STENCILUTILS_H

#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <bits/stdint-intn.h>
#include <functional>

namespace mlir {
namespace stencil {

/// Helper method that computes the minimum and maximum index
int64_t min(int64_t x, int64_t y);
int64_t max(int64_t x, int64_t y);

/// Helper method to compute element-wise index computations
Index applyFunElementWise(ArrayRef<int64_t> x, ArrayRef<int64_t> y,
                          std::function<int64_t(int64_t, int64_t)> fun);

} // namespace stencil
} // namespace mlir

#endif // DIALECT_STENCIL_STENCILUTILS_H
