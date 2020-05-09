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
int64_t minimum(int64_t x, int64_t y);
int64_t maximum(int64_t x, int64_t y);

/// Helper method to compute element-wise index computations
Index mapFunctionToIdxPair(ArrayRef<int64_t> x, ArrayRef<int64_t> y,
                           std::function<int64_t(int64_t, int64_t)> fun);

/// Helper methos to convert between index variables and attributes
ArrayAttr convertIndexToAttr(ArrayRef<int64_t> index, MLIRContext *context);
Index convertAttrToIndex(ArrayAttr attr);
Index convertAttrToIndex(Optional<ArrayAttr> attr);

/// Acces stencil field and temporary properties
Type getElementType(Value value);
ArrayRef<int64_t> getShape(Value value);

/// Check if an operation has a given type
template <typename TOp>
unsigned hasOpType(Operation *op) {
  return isa_and_nonnull<TOp>(op);
}

} // namespace stencil
} // namespace mlir

#endif // DIALECT_STENCIL_STENCILUTILS_H
