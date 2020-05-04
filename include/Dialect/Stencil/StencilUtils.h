#ifndef DIALECT_STENCIL_STENCILUTILS_H
#define DIALECT_STENCIL_STENCILUTILS_H

#include "mlir/IR/Attributes.h"

namespace mlir {
namespace stencil {

// Convert a vector to an attribute
ArrayAttr convertVecToAttr(ArrayRef<int64_t> vector, MLIRContext *context);

// Convert an attribute to a vector
SmallVector<int64_t, 3> convertAttrToVec(ArrayAttr attr);
SmallVector<int64_t, 3> convertAttrToVec(Optional<ArrayAttr> attr);

} // namespace stencil
} // namespace mlir

#endif // DIALECT_STENCIL_STENCILUTILS_H
