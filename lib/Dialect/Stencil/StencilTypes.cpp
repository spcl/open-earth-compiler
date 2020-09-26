#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

using namespace mlir;
using namespace stencil;

namespace mlir {
namespace stencil {
namespace detail {

struct GridTypeStorage : public TypeStorage {
  GridTypeStorage(Type elementTy, size_t size, const int64_t *shape)
      : TypeStorage(), elementType(elementTy), size(size), shape(shape) {}

  /// Hash key used for uniquing
  using KeyTy = std::pair<Type, ArrayRef<int64_t>>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, getShape());
  }

  Type getElementType() const { return elementType; }
  ArrayRef<int64_t> getShape() const { return {shape, size}; }

  Type elementType;
  const size_t size;
  const int64_t *shape;
};

struct FieldTypeStorage : public GridTypeStorage {
  using GridTypeStorage::GridTypeStorage;

  /// Construction
  static FieldTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    // Copy the allocation into the bump pointer.
    ArrayRef<int64_t> shape = allocator.copyInto(key.second);

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
    ArrayRef<int64_t> shape = allocator.copyInto(key.second);

    return new (allocator.allocate<TempTypeStorage>())
        TempTypeStorage(key.first, shape.size(), shape.data());
  }
};

struct ResultTypeStorage : public TypeStorage {
  ResultTypeStorage(Type resultType) : TypeStorage(), resultType(resultType) {}

  /// Hash key used for uniquing
  using KeyTy = Type;

  bool operator==(const KeyTy &key) const { return key == resultType; }

  Type getResultType() const { return resultType; }

  /// Construction
  static ResultTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<ResultTypeStorage>()) ResultTypeStorage(key);
  }

  Type resultType;
};

} // namespace detail
} // namespace stencil
} // namespace mlir

//===----------------------------------------------------------------------===//
// GridType
//===----------------------------------------------------------------------===//

constexpr int64_t GridType::kDynamicDimension;
constexpr int64_t GridType::kScalarDimension;

bool GridType::classof(Type type) { return type.isa<FieldType, TempType>(); }

Type GridType::getElementType() const {
  return static_cast<ImplType *>(impl)->getElementType();
}

ArrayRef<int64_t> GridType::getShape() const {
  return static_cast<ImplType *>(impl)->getShape();
}

int64_t GridType::getRank() const { return getShape().size(); }

int64_t GridType::hasDynamicShape() const {
  return llvm::all_of(getShape(), [](int64_t size) {
    return size == kDynamicDimension || size == kScalarDimension;
  });
}

int64_t GridType::hasStaticShape() const {
  return llvm::none_of(getShape(),
                       [](int64_t size) { return size == kDynamicDimension; });
}

SmallVector<bool, 3> GridType::getAllocation() const {
  SmallVector<bool, 3> result;
  result.resize(getRank());
  llvm::transform(getShape(), result.begin(),
                  [](int64_t x) { return x != kScalarDimension; });
  return result;
}

SmallVector<int64_t, 3> GridType::getMemRefShape() const {
  SmallVector<int64_t, 3> result;
  for (auto size : llvm::reverse(getShape())) {
    switch (size) {
    case (kDynamicDimension):
      result.push_back(ShapedType::kDynamicSize);
      break;
    case (kScalarDimension):
      break;
    default:
      result.push_back(size);
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// FieldType
//===----------------------------------------------------------------------===//

FieldType FieldType::get(Type elementType, llvm::ArrayRef<int64_t> shape) {
  return Base::get(elementType.getContext(), elementType, shape);
}

//===----------------------------------------------------------------------===//
// TempType
//===----------------------------------------------------------------------===//

TempType TempType::get(Type elementType, llvm::ArrayRef<int64_t> shape) {
  return Base::get(elementType.getContext(), elementType, shape);
}

//===----------------------------------------------------------------------===//
// ResultType
//===----------------------------------------------------------------------===//

ResultType ResultType::get(Type resultType) {
  return Base::get(resultType.getContext(), resultType);
}

Type ResultType::getResultType() const { return getImpl()->getResultType(); }