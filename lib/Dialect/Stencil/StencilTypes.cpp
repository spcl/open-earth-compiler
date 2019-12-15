#include "Dialect/Stencil/StencilTypes.h"

using namespace mlir;
using namespace mlir::stencil;

struct mlir::stencil::FieldTypeStorage : public TypeStorage {
  /// Underlying Key type to transport the payload needed to construct a custom
  /// type in a generic way.
  using Key = std::pair<ArrayRef<int64_t>, Type>;

  /// `KeyTy` is a necessary typename hook for MLIR's custom type unique'ing.
  using KeyTy = Key;

  /// Construction in the `llvm::BumpPtrAllocator` given a key.
  static FieldTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    // Copy the shape into the bump pointer.
    ArrayRef<int64_t> shape = allocator.copyInto(key.first);

    // Initialize the memory using placement new.
    return new (allocator.allocate<FieldTypeStorage>())
        FieldTypeStorage(shape.size(), shape.data(), key.second);
  }

  /// Equality operator for hashing.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getShape(), elementType);
  }

  /// Return the type of the field elements.
  Type getElementType() const { return elementType; }

  /// Return the shape of the field.
  ArrayRef<int64_t> getShape() const { return {shapeElems, shapeSize}; }

private:
  FieldTypeStorage(size_t shapeSize, const int64_t *shape, Type elementType)
      : shapeSize(shapeSize), shapeElems(shape), elementType(elementType) {}

  /// Number of shape dimensions.
  const size_t shapeSize;
  /// Shape size in each dimension.
  const int64_t *shapeElems;
  /// Type of the field elements.
  Type elementType;
};

FieldType FieldType::get(mlir::MLIRContext *context, mlir::Type elementType,
                         llvm::ArrayRef<int64_t> shape) {
  assert(shape.size() == 3 && "field shape must have 3 dimensions");
  assert((elementType.isInteger(64) || elementType.isF64()) &&
         "fields only support i64 and f64 elements");
  return Base::get(context, StencilTypes::Field, shape, elementType);
}

Type FieldType::getElementType() { return getImpl()->getElementType(); }

ArrayRef<int64_t> FieldType::getShape() { return getImpl()->getShape(); }

struct mlir::stencil::ViewTypeStorage : public TypeStorage {
  /// Underlying Key type to transport the payload needed to construct a custom
  /// type in a generic way.
  using Key = std::pair<ArrayRef<int64_t>, Type>;

  /// `KeyTy` is a necessary typename hook for MLIR's custom type unique'ing.
  using KeyTy = Key;

  /// Construction in the `llvm::BumpPtrAllocator` given a key.
  static ViewTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    // Copy the shape into the bump pointer.
    ArrayRef<int64_t> shape = allocator.copyInto(key.first);

    // Initialize the memory using placement new.
    return new (allocator.allocate<ViewTypeStorage>())
        ViewTypeStorage(shape.size(), shape.data(), key.second);
  }

  /// Equality operator for hashing.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getShape(), elementType);
  }

  /// Return the type of the field elements.
  Type getElementType() const { return elementType; }

  /// Return the shape of the field.
  ArrayRef<int64_t> getShape() const { return {shapeElems, shapeSize}; }

private:
  ViewTypeStorage(size_t shapeSize, const int64_t *shape, Type elementType)
      : shapeSize(shapeSize), shapeElems(shape), elementType(elementType) {}

  /// Number of shape dimensions.
  const size_t shapeSize;
  /// Shape size in each dimension.
  const int64_t *shapeElems;
  /// Type of the field elements.
  Type elementType;
};

ViewType ViewType::get(mlir::MLIRContext *context, mlir::Type elementType,
                       llvm::ArrayRef<int64_t> shape) {
  assert(shape.size() == 3 && "view shape must have 3 dimensions");
  assert((elementType.isInteger(64) || elementType.isF64()) &&
         "views only support i64 and f64 elements");
  return Base::get(context, StencilTypes::View, shape, elementType);
}

Type ViewType::getElementType() { return getImpl()->getElementType(); }

ArrayRef<int64_t> ViewType::getShape() { return getImpl()->getShape(); }
