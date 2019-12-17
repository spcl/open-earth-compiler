#ifndef MLIR_DIALECT_STENCIL_STENCILTYPES_H
#define MLIR_DIALECT_STENCIL_STENCILTYPES_H

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace stencil {

namespace StencilTypes {
enum Kind {
  Field = Type::FIRST_STENCIL_TYPE,
  View,
  LAST_USED_STENCIL_TYPE = View
};
}

namespace StencilStorage {
  enum Allocation : unsigned { IJK, IJ, IK, JK, I, J, K };
}

struct FieldTypeStorage;
class FieldType : public Type::TypeBase<FieldType, Type, FieldTypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;

  /// Construction hook.
  static FieldType get(MLIRContext *context, Type elementType,
                       StencilStorage::Allocation allocation);

  /// Used to implement LLVM-style casts.
  static bool kindof(unsigned kind) { return kind == StencilTypes::Field; }

  /// Return the type of the field elements.
  Type getElementType();
  /// Return the allocation of the field.
  StencilStorage::Allocation getAllocation();
};

struct ViewTypeStorage;
class ViewType : public Type::TypeBase<ViewType, Type, ViewTypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;

  /// Construction hook.
  static ViewType get(MLIRContext *context, Type elementType,
                      StencilStorage::Allocation allocation);

  /// Used to implement LLVM-style casts.
  static bool kindof(unsigned kind) { return kind == StencilTypes::View; }

  /// Return the type of the field elements.
  Type getElementType();
  /// Return the allocation of the field.
  StencilStorage::Allocation getAllocation();
};

} // namespace stencil
} // namespace mlir

#endif // MLIR_DIALECT_STENCIL_STENCILTYPES_H
