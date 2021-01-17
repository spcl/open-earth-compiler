#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace stencil;

namespace {

/// This class computes the access sets for all store operations
class AccessSets {
public:
  AccessSets(Operation *op) {
    op->walk([&](StoreOp storeOp) {
      // Search all load operations
      SmallVector<Operation *, 10> queue = {storeOp};
      while (!queue.empty()) {
        auto curr = queue.back();
        queue.pop_back();

        // Add field to the access set
        if (auto loadOp = dyn_cast<LoadOp>(curr)) {
          accessSets[storeOp].insert(loadOp.field());
          continue;
        }

        // Search possible access sets
        for (auto operand : curr->getOperands()) {
          if (auto definingOp = operand.getDefiningOp()) {
            queue.push_back(definingOp);
          }
        }
      }
    });
  }

  /// Return true if the access sets of the two store operations overlap
  bool areSetsOverlapping(StoreOp storeOp1, StoreOp storeOp2) {
    for (auto value : accessSets[storeOp1]) {
      if (accessSets[storeOp2].count(value) != 0)
        return true;
    }
    return false;
  }

private:
  llvm::DenseMap<Operation *, llvm::DenseSet<Value>> accessSets;
};

struct ShapeOverlapPass : public ShapeOverlapPassBase<ShapeOverlapPass> {
  void runOnFunction() override;

protected:
  bool areRangesOverlapping(ShapeOp shapeOp1, ShapeOp shapeOp2);
  SmallVector<Value, 10> computeGroupValues(OpBuilder b, int dim, int64_t from,
                                            int64_t to, ArrayRef<int64_t> lb,
                                            ArrayRef<int64_t> ub,
                                            ArrayRef<StoreOp> group);
  SmallVector<Value, 10> splitGroupDimension(OpBuilder b, int dim,
                                             ArrayRef<int64_t> lb,
                                             ArrayRef<int64_t> ub,
                                             ArrayRef<StoreOp> group);
};

// Two shapes overlap if at least half of the bounds are overlapping
bool ShapeOverlapPass::areRangesOverlapping(ShapeOp shapeOp1,
                                            ShapeOp shapeOp2) {
  auto count = count_if(zip(shapeOp1.getLB(), shapeOp2.getLB()),
                        [](std::tuple<int64_t, int64_t> x) {
                          return std::get<0>(x) == std::get<1>(x);
                        }) +
               count_if(zip(shapeOp1.getUB(), shapeOp2.getUB()),
                        [](std::tuple<int64_t, int64_t> x) {
                          return std::get<0>(x) == std::get<1>(x);
                        });
  return count >= kIndexSize;
}

SmallVector<Value, 10> ShapeOverlapPass::computeGroupValues(
    OpBuilder b, int dim, int64_t from, int64_t to, ArrayRef<int64_t> lb,
    ArrayRef<int64_t> ub, ArrayRef<StoreOp> group) {
  // Iterate all store operations of the group
  SmallVector<StoreOp, 10> subGroup;
  SmallVector<size_t, 10> subGroupIndexes;
  for (auto en : llvm::enumerate(group)) {
    auto storeOp = en.value();
    auto shapeOp = cast<ShapeOp>(storeOp.getOperation());
    if (shapeOp.getLB()[dim] <= from && shapeOp.getUB()[dim] >= to) {
      subGroup.push_back(storeOp);
      subGroupIndexes.push_back(en.index());
    }
  }

  // Split the subgroup recursively
  auto subTempValues = splitGroupDimension(b, dim - 1, lb, ub, subGroup);

  // Return the resulting temp values or null otherwise
  SmallVector<Value, 10> tempValues(group.size(), nullptr);
  for (auto en : llvm::enumerate(subGroupIndexes)) {
    tempValues[en.value()] = subTempValues[en.index()];
  }
  return tempValues;
}

SmallVector<Value, 10> ShapeOverlapPass::splitGroupDimension(
    OpBuilder b, int dim, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub,
    ArrayRef<StoreOp> group) {
  assert(dim < kIndexSize &&
         "expected dimension to be lower than the index size");
  // Return the group temporary values if the dimension is smaller than zero
  if (dim < 0) {
    SmallVector<Value, 10> tempValues;
    llvm::transform(group, std::back_inserter(tempValues),
                    [](StoreOp storeOp) { return storeOp.temp(); });
    return tempValues;
  }

  // Compute the bounds of all subdomains
  SmallVector<int64_t, 10> limits = {lb[dim], ub[dim]};
  for (auto storeOp : group) {
    auto shapeOp = cast<ShapeOp>(storeOp.getOperation());
    limits.push_back(shapeOp.getLB()[dim]);
    limits.push_back(shapeOp.getUB()[dim]);
  }
  std::sort(limits.begin(), limits.end());
  limits.erase(std::unique(limits.begin(), limits.end()), limits.end());

  // Compute the lower and upper bounds of all intervals except for the last
  SmallVector<int64_t, 10> lowerBounds;
  SmallVector<int64_t, 10> upperBounds;
  for (int32_t i = 1, e = limits.size(); i != e; ++i) {
    lowerBounds.push_back(limits[i - 1]);
    upperBounds.push_back(limits[i]);
  }

  // Initialize the temporary values to the values stored in the last intervall
  auto tempValues = computeGroupValues(b, dim, lowerBounds.back(),
                                       upperBounds.back(), lb, ub, group);
  lowerBounds.pop_back();
  upperBounds.pop_back();

  // Introduce combine operations in backward order
  while (!lowerBounds.empty()) {
    auto currValues = computeGroupValues(b, dim, lowerBounds.back(),
                                         upperBounds.back(), lb, ub, group);

    // Compute the indexes of the lower upper and extra values
    SmallVector<int, 10> lower, lowerext, upperext;
    for (auto en : llvm::enumerate(currValues)) {
      if (en.value() && tempValues[en.index()]) {
        lower.push_back(en.index());
      } else if (en.value()) {
        lowerext.push_back(en.index());
      } else if (tempValues[en.index()]) {
        upperext.push_back(en.index());
      }
    }

    // Compute the result types
    SmallVector<Type, 10> resultTypes;
    llvm::transform(lower, std::back_inserter(resultTypes),
                    [&](int64_t x) { return currValues[x].getType(); });
    llvm::transform(lowerext, std::back_inserter(resultTypes),
                    [&](int64_t x) { return currValues[x].getType(); });
    llvm::transform(upperext, std::back_inserter(resultTypes),
                    [&](int64_t x) { return tempValues[x].getType(); });

    // Compute the lower upper and extra operands
    SmallVector<Value, 10> lowerOperands, upperOperands, lowerextOperands,
        upperextOperands;
    llvm::transform(lower, std::back_inserter(lowerOperands),
                    [&](int64_t x) { return currValues[x]; });
    llvm::transform(lower, std::back_inserter(upperOperands),
                    [&](int64_t x) { return tempValues[x]; });
    llvm::transform(lowerext, std::back_inserter(lowerextOperands),
                    [&](int64_t x) { return currValues[x]; });
    llvm::transform(upperext, std::back_inserter(upperextOperands),
                    [&](int64_t x) { return tempValues[x]; });

    // Compute the fused location
    SmallVector<Location, 10> locs;
    llvm::transform(group, std::back_inserter(locs),
                    [](StoreOp storeOp) { return storeOp->getLoc(); });

    // Create a combine operation
    auto combineOp = b.create<stencil::CombineOp>(
        locs.empty() ? b.getUnknownLoc() : b.getFusedLoc(locs), resultTypes,
        dim, upperBounds.back(), lowerOperands, upperOperands, lowerextOperands,
        upperextOperands, nullptr, nullptr);

    // Update the temporary values
    unsigned resultIdx = 0;
    for (int64_t x : lower)
      tempValues[x] = combineOp->getResult(resultIdx++);
    for (int64_t x : lowerext)
      tempValues[x] = combineOp->getResult(resultIdx++);
    for (int64_t x : upperext)
      tempValues[x] = combineOp->getResult(resultIdx++);

    lowerBounds.pop_back();
    upperBounds.pop_back();
  }

  return tempValues;
}

} // namespace

void ShapeOverlapPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!stencil::StencilDialect::isStencilProgram(funcOp))
    return;

  // Compute the extent analysis
  AccessSets &sets = getAnalysis<AccessSets>();

  // Walk all store operations
  SmallVector<SmallVector<StoreOp, 10>, 4> groupList;
  funcOp->walk([&](StoreOp storeOp1) {
    // Check if operation overlaps with one of the groups
    for (auto &group : groupList) {
      if (llvm::any_of(group, [&](StoreOp storeOp2) {
            return areRangesOverlapping(storeOp2, storeOp1) &&
                   sets.areSetsOverlapping(storeOp2, storeOp1);
          })) {
        group.push_back(storeOp1);
        return;
      }
    }
    // Otherwise add another group
    groupList.push_back({storeOp1});
  });

  // Split all groups
  for (auto &group : groupList) {
    // Search the first store operation
    auto storeOp = *std::min_element(
        group.begin(), group.end(), [](StoreOp storeOp1, StoreOp storeOp2) {
          return storeOp1->isBeforeInBlock(storeOp2);
        });
    OpBuilder b(storeOp);

    // Compute the bounding box of all store operation shapes
    auto shapeOp = cast<ShapeOp>(storeOp.getOperation());
    auto lb = shapeOp.getLB();
    auto ub = shapeOp.getUB();
    for (ShapeOp shapeOp : group) {
      lb = applyFunElementWise(lb, shapeOp.getLB(), min);
      ub = applyFunElementWise(ub, shapeOp.getUB(), max);
    }

    // Split the group dimensions recursively
    auto tempValues = splitGroupDimension(b, kIndexSize - 1, lb, ub, group);

    // Update the store operands
    for (auto en : llvm::enumerate(group)) {
      en.value()->setOperand(0, tempValues[en.index()]);
    }
  }
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createShapeOverlapPass() {
  return std::make_unique<ShapeOverlapPass>();
}
