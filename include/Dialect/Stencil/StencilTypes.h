#ifndef MLIR_DIALECT_STENCIL_STENCILTYPES_H
#define MLIR_DIALECT_STENCIL_STENCILTYPES_H

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include <bits/stdint-intn.h>
#include <cstdint>

namespace mlir {
namespace stencil {

namespace StencilTypes {
enum Kind {
  Field = Type::FIRST_STENCIL_TYPE,
  View,
  LAST_USED_STENCIL_TYPE = View
};
}

struct FieldTypeStorage;
class FieldType : public Type::TypeBase<FieldType, Type, FieldTypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;

  /// Construction hook.
  static FieldType get(MLIRContext *context, Type elementType,
                       ArrayRef<int> dimensions);

  /// Used to implement LLVM-style casts.
  static bool kindof(unsigned kind) { return kind == StencilTypes::Field; }

  /// Return the type of the field elements.
  Type getElementType();
  /// Return the allocated dimensions of the field.
  ArrayRef<int> getDimensions();
};

struct ViewTypeStorage;
class ViewType : public Type::TypeBase<ViewType, Type, ViewTypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;

  /// Construction hook.
  static ViewType get(MLIRContext *context, Type elementType,
                      ArrayRef<int> dimensions);

  /// Used to implement LLVM-style casts.
  static bool kindof(unsigned kind) { return kind == StencilTypes::View; }

  /// Return the type of the view elements.
  Type getElementType();
  /// Return the allocated dimensions of the view.
  ArrayRef<int> getDimensions();
};

} // namespace stencil
} // namespace mlir

#endif // MLIR_DIALECT_STENCIL_STENCILTYPES_H
