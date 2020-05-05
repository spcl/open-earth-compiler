#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace stencil;

namespace mlir {
namespace stencil {
namespace detail {

struct GridTypeStorage : public TypeStorage {
  GridTypeStorage(Type elementTy, size_t size, const int32_t *shape)
      : TypeStorage(), elementType(elementTy), size(size), shape(shape) {}

  /// Hash key used for uniquing
  using KeyTy = std::pair<Type, ArrayRef<int32_t>>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, getShape());
  }

  Type getElementType() const { return elementType; }
  ArrayRef<int32_t> getShape() const { return {shape, size}; }

  Type elementType;
  const size_t size;
  const int32_t *shape;
};

struct FieldTypeStorage : public GridTypeStorage {
  using GridTypeStorage::GridTypeStorage;

  /// Construction
  static FieldTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    // Copy the allocation into the bump pointer.
    ArrayRef<int32_t> shape = allocator.copyInto(key.second);

    return new (allocator.allocate<FieldTypeStorage>())
        FieldTypeStorage(key.first, shape.size(), shape.data());
  }
};

struct TempTypeStorage : public GridTypeStorage {
  using GridTypeStorage::GridTypeStorage;

  /// Construction
  static TempTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    // Copy the allocation into the bump pointer.
    ArrayRef<int32_t> shape = allocator.copyInto(key.second);

    return new (allocator.allocate<TempTypeStorage>())
        TempTypeStorage(key.first, shape.size(), shape.data());
  }
};

} // namespace detail
} // namespace stencil
} // namespace mlir

//===----------------------------------------------------------------------===//
// GridType
//===----------------------------------------------------------------------===//

constexpr int32_t GridType::kDynamicSize;
constexpr int32_t GridType::kScalarDim;

Type GridType::getElementType() const {
  return static_cast<ImplType *>(impl)->getElementType();
}

ArrayRef<int> GridType::getShape() const {
  return static_cast<ImplType *>(impl)->getShape();
}

//===----------------------------------------------------------------------===//
// FieldType
//===----------------------------------------------------------------------===//

FieldType FieldType::get(MLIRContext *context, Type elementType,
                         llvm::ArrayRef<int> shape) {
  return Base::get(context, StencilTypes::Field, elementType, shape);
}

//===----------------------------------------------------------------------===//
// TempType
//===----------------------------------------------------------------------===//

TempType TempType::get(MLIRContext *context, Type elementType,
                       llvm::ArrayRef<int> shape) {
  return Base::get(context, StencilTypes::Temp, elementType, shape);
}
