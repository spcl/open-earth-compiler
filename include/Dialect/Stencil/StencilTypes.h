#ifndef MLIR_DIALECT_STENCIL_STENCILTYPES_H
#define MLIR_DIALECT_STENCIL_STENCILTYPES_H

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace stencil {

enum StencilTypes {
  Field = Type::FIRST_STENCIL_TYPE,
  View,
  LAST_USED_STENCIL_TYPE = View
};

struct FieldTypeStorage;
class FieldType : public Type::TypeBase<FieldType, Type, FieldTypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;

  /// Construction hook.
  ///
  /// Create a field type of given `shape` containing elements of type
  /// `elementType`.
  ///
  /// Note: `shape` must contain 3 elements, -1 being used to specify an unknown
  /// size.
  static FieldType get(MLIRContext *context, Type elementType,
                       ArrayRef<int64_t> shape);

  /// Used to implement LLVM-style casts.
  static bool kindof(unsigned kind) { return kind == StencilTypes::Field; }

  /// Return the type of the field elements.
  Type getElementType();
  /// Return the shape of the field.
  ArrayRef<int64_t> getShape();
};

struct ViewTypeStorage;
class ViewType : public Type::TypeBase<ViewType, Type, ViewTypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;

  /// Construction hook.
  ///
  /// Create a field type of given `shape` containing elements of type
  /// `elementType`.
  ///
  /// Note: `shape` must contain 3 elements, -1 being used to specify an unknown
  /// size.
  static ViewType get(MLIRContext *context, Type elementType,
                      ArrayRef<int64_t> shape);

  /// Used to implement LLVM-style casts.
  static bool kindof(unsigned kind) { return kind == StencilTypes::View; }

  /// Return the type of the field elements.
  Type getElementType();
  /// Return the shape of the field.
  ArrayRef<int64_t> getShape();
};

} // namespace stencil
} // namespace mlir

#endif // MLIR_DIALECT_STENCIL_STENCILTYPES_H
