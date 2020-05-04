#ifndef DIALECT_STENCIL_STENCILUTILS_H
#define DIALECT_STENCIL_STENCILUTILS_H

#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace stencil {

// Convert a vector to an attribute
ArrayAttr convertVecToAttr(ArrayRef<int64_t> vector, MLIRContext *context);

// Convert an attribute to a vector
SmallVector<int64_t, 3> convertAttrToVec(ArrayAttr attr);
SmallVector<int64_t, 3> convertAttrToVec(Optional<ArrayAttr> attr);

// Acces stencil field and temporary properties
Type getElementType(Value value);
ArrayRef<int> getDimensions(Value value);

// Return true if the operation has the given type
template <typename TOp>
unsigned hasOpType(Operation *op) {
  return isa_and_nonnull<TOp>(op);
}

} // namespace stencil
} // namespace mlir

#endif // DIALECT_STENCIL_STENCILUTILS_H
