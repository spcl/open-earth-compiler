#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "PassDetail.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <bits/stdint-intn.h>

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
  SmallVector<Value, 10> computeGroupValues(int dim, int64_t lb, int64_t ub,
                                            ArrayRef<StoreOp> group);
  void splitGroupDimension(int dim, ArrayRef<StoreOp> group);
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

SmallVector<Value, 10>
ShapeOverlapPass::computeGroupValues(int dim, int64_t lb, int64_t ub,
                                     ArrayRef<StoreOp> group) {
  // Iterate all store operations of the group
  SmallVector<Value, 10> tempValues;
  for (auto storeOp : group) {
    auto shapeOp = cast<ShapeOp>(storeOp.getOperation());
    if (shapeOp.getLB()[dim] <= lb && shapeOp.getUB()[dim] >= ub) {
      tempValues.push_back(storeOp.temp());
    } else {
      tempValues.push_back(nullptr);
    }
  }
  return tempValues;
}

void ShapeOverlapPass::splitGroupDimension(int dim, ArrayRef<StoreOp> group) {
  assert(dim < kIndexSize &&
         "expected dimension to be lower than the index size");
  // Compute the bounds of all subdomains
  SmallVector<int64_t, 10> limits;
  for (auto storeOp : group) {
    auto shapeOp = cast<ShapeOp>(storeOp.getOperation());
    limits.push_back(shapeOp.getLB()[dim]);
    limits.push_back(shapeOp.getUB()[dim]);
  }
  std::sort(limits.begin(), limits.end());
  limits.erase(std::unique(limits.begin(), limits.end()), limits.end());

  // Setup the op builder
  auto storeOp = *std::min_element(group.begin(), group.end(),
                                   [](StoreOp storeOp1, StoreOp storeOp2) {
                                     return storeOp1->isBeforeInBlock(storeOp2);
                                   });
  OpBuilder b(storeOp);

  // Compute the lower and upper bounds of all intervals except for the last
  SmallVector<int64_t, 10> lowerBounds;
  SmallVector<int64_t, 10> upperBounds;
  for (int32_t i = 1, e = limits.size(); i != e; ++i) {
    lowerBounds.push_back(limits[i - 1]);
    upperBounds.push_back(limits[i]);
  }

  // Initialize the temporary values to the values stored in the last intervall
  auto tempValues =
      computeGroupValues(dim, lowerBounds.back(), upperBounds.back(), group);
  lowerBounds.pop_back();
  upperBounds.pop_back();

  // Introduce combine operations in backward order
  while (!lowerBounds.empty()) {
    auto currValues =
        computeGroupValues(dim, lowerBounds.back(), upperBounds.back(), group);

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

    // Create a combine operation
    auto combineOp = b.create<stencil::CombineOp>(
        storeOp.getLoc(), resultTypes, dim, upperBounds.back(), lowerOperands,
        upperOperands, lowerextOperands, upperextOperands, nullptr, nullptr);

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

  // Update the store operands
  for(auto en : llvm::enumerate(group)) {
    en.value()->setOperand(0, tempValues[en.index()]);
  }
}

} // namespace

void ShapeOverlapPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!stencil::StencilDialect::isStencilProgram(funcOp))
    return;

  // Compute the extent analysis
  AccessSets &sets = getAnalysis<AccessSets>();

  // TODO check before shape inference

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
    splitGroupDimension(0, group);
  }
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createShapeOverlapPass() {
  return std::make_unique<ShapeOverlapPass>();
}
