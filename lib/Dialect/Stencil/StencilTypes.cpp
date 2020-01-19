#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
//#include <bits/stdint-intn.h>
#include <algorithm>
#include <cstddef>
#include <set>

using namespace mlir;
using namespace mlir::stencil;

//===----------------------------------------------------------------------===//
// FieldType
//===----------------------------------------------------------------------===//

struct mlir::stencil::FieldTypeStorage : public TypeStorage {
  using Key = std::pair<ArrayRef<int>, Type>;
  using KeyTy = Key;

  static FieldTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    // Copy the allocation into the bump pointer.
    ArrayRef<int> dimensions = allocator.copyInto(key.first);

    // Initialize the memory using placement new.
    return new (allocator.allocate<FieldTypeStorage>())
        FieldTypeStorage(dimensions.size(), dimensions.data(), key.second);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getDimensions(), elementType);
  }

  Type getElementType() const { return elementType; }
  ArrayRef<int> getDimensions() const { return {dimensions, size}; }

private:
  FieldTypeStorage(size_t size, const int *dimensions, Type elementType)
      : size(size), dimensions(dimensions), elementType(elementType) {}

  const size_t size;
  const int *dimensions;
  Type elementType;
};

FieldType FieldType::get(mlir::MLIRContext *context, mlir::Type elementType,
                         llvm::ArrayRef<int> dimensions) {
  assert((elementType.isF32() || elementType.isF64()) &&
         "supporting only f32 and f64 elements");
  assert(dimensions.size() > 0 && dimensions.size() <= 3 &&
         "expected up to three dimensions");
  // Sort the dimensions
  std::vector<int> temp = dimensions.vec();
  std::sort(temp.begin(), temp.end());
  auto last = std::unique(temp.begin(), temp.end());
  assert(temp.end() == last &&
         "expected list of unique dimensions identifiers");
  assert(temp.front() >= 0 && temp.back() < 3 &&
         "dimension identifiers are out of range");
  return Base::get(context, StencilTypes::Field, temp, elementType);
}

Type FieldType::getElementType() { return getImpl()->getElementType(); }

ArrayRef<int> FieldType::getDimensions() { return getImpl()->getDimensions(); }

//===----------------------------------------------------------------------===//
// ViewType
//===----------------------------------------------------------------------===//

struct mlir::stencil::ViewTypeStorage : public TypeStorage {
  using Key = std::pair<ArrayRef<int>, Type>;
  using KeyTy = Key;

  static ViewTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    // Copy the allocation into the bump pointer.
    ArrayRef<int> dimensions = allocator.copyInto(key.first);

    // Initialize the memory using placement new.
    return new (allocator.allocate<ViewTypeStorage>())
        ViewTypeStorage(dimensions.size(), dimensions.data(), key.second);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getDimensions(), elementType);
  }

  Type getElementType() const { return elementType; }
  ArrayRef<int> getDimensions() const { return {dimensions, size}; }

private:
  ViewTypeStorage(size_t size, const int *dimensions, Type elementType)
      : size(size), dimensions(dimensions), elementType(elementType) {}

  const size_t size;
  const int *dimensions;
  Type elementType;
};

ViewType ViewType::get(mlir::MLIRContext *context, mlir::Type elementType,
                       llvm::ArrayRef<int> dimensions) {
  assert((elementType.isF32() || elementType.isF64()) &&
         "supporting only f32 and f64 elements");
  assert(dimensions.size() > 0 && dimensions.size() <= 3 &&
         "expected up to three dimensions");
  // Sort the dimensions
  std::vector<int> temp = dimensions.vec();
  std::sort(temp.begin(), temp.end());
  auto last = std::unique(temp.begin(), temp.end());
  assert(temp.end() == last &&
         "expected list of unique dimensions identifiers");
  assert(temp.front() >= 0 && temp.back() < 3 &&
         "dimension identifiers are out of range");
  return Base::get(context, StencilTypes::View, temp, elementType);
}

Type ViewType::getElementType() { return getImpl()->getElementType(); }

ArrayRef<int> ViewType::getDimensions() { return getImpl()->getDimensions(); }
