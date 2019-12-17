#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace mlir::stencil;

struct mlir::stencil::FieldTypeStorage : public TypeStorage {
  /// Underlying Key type to transport the payload needed to construct a custom
  /// type in a generic way.
  using Key = std::pair<unsigned, Type>;

  /// `KeyTy` is a necessary typename hook for MLIR's custom type unique'ing.
  using KeyTy = Key;

  /// Construction in the `llvm::BumpPtrAllocator` given a key.
  static FieldTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<FieldTypeStorage>()) FieldTypeStorage(
        static_cast<StencilStorage::Allocation>(key.first), key.second);
  }

  /// Equality operator for hashing.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getAllocation(), elementType);
  }

  /// Return the type of the field elements.
  Type getElementType() const { return elementType; }

  /// Return the shape of the field.
  StencilStorage::Allocation getAllocation() const { return allocation; }

private:
  FieldTypeStorage(StencilStorage::Allocation allocation, Type elementType)
      : allocation(allocation), elementType(elementType) {}

  /// Allocation of the storage.
  StencilStorage::Allocation allocation;
  /// Type of the field elements.
  Type elementType;
};

FieldType FieldType::get(mlir::MLIRContext *context, mlir::Type elementType,
                         StencilStorage::Allocation allocation) {
  assert((elementType.isF32() || elementType.isF64()) &&
         "fields only support f32 and f64 elements");
  return Base::get(context, StencilTypes::Field, allocation, elementType);
}

Type FieldType::getElementType() { return getImpl()->getElementType(); }

StencilStorage::Allocation FieldType::getAllocation() {
  return getImpl()->getAllocation();
}

struct mlir::stencil::ViewTypeStorage : public TypeStorage {
  /// Underlying Key type to transport the payload needed to construct a custom
  /// type in a generic way.
  using Key = std::pair<unsigned, Type>;

  /// `KeyTy` is a necessary typename hook for MLIR's custom type unique'ing.
  using KeyTy = Key;

  /// Construction in the `llvm::BumpPtrAllocator` given a key.
  static ViewTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<ViewTypeStorage>()) ViewTypeStorage(
        static_cast<StencilStorage::Allocation>(key.first), key.second);
  }

  /// Equality operator for hashing.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getAllocation(), elementType);
  }

  /// Return the type of the field elements.
  Type getElementType() const { return elementType; }

  /// Return the shape of the field.
  StencilStorage::Allocation getAllocation() const { return allocation; }

private:
  ViewTypeStorage(StencilStorage::Allocation allocation, Type elementType)
      : allocation(allocation), elementType(elementType) {}

  /// Allocation of the storage.
  StencilStorage::Allocation allocation;
  /// Type of the field elements.
  Type elementType;
};

ViewType ViewType::get(mlir::MLIRContext *context, mlir::Type elementType,
                       StencilStorage::Allocation allocation) {
  assert((elementType.isF32() || elementType.isF64()) &&
         "views only support f32 and f64 elements");
  return Base::get(context, StencilTypes::View, allocation, elementType);
}

Type ViewType::getElementType() { return getImpl()->getElementType(); }

StencilStorage::Allocation ViewType::getAllocation() {
  return getImpl()->getAllocation();
}
