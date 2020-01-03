#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include <cstddef>
#include <limits>

using namespace mlir;
using namespace stencil;

namespace {

/// This class computes for every stencil apply operand
/// the minimal bounding box containing all access offsets
class AccessExtents {
  // This struct stores the positive and negative extends
  struct Extent {
    SmallVector<int64_t, 3> negative;
    SmallVector<int64_t, 3> positive;
  };

public:
  AccessExtents(Operation *op) {
    // Walk all apply ops of the stencil program
    op->walk([&](stencil::ApplyOp applyOp) {
      auto operation = applyOp.getOperation();
      // Compute mapping between operands and block arguments
      llvm::DenseMap<Value, Value> argumentsToOperands;
      for (size_t i = 0, e = applyOp.operands().size(); i != e; ++i) {
        argumentsToOperands[applyOp.getBody()->getArgument(i)] =
            applyOp.operands()[i];
      }
      // Walk the access ops and update the extent
      applyOp.walk([&](stencil::AccessOp accessOp) {
        auto offset = accessOp.getOffset();
        auto argument = accessOp.getOperand();
        if (extents[operation].count(argumentsToOperands[argument]) == 0) {
          // Initialize the extents with the current offset
          extents[operation][argumentsToOperands[argument]].negative = offset;
          extents[operation][argumentsToOperands[argument]].positive = offset;
        } else {
          // Extend the extents with the current offset
          auto &extent = extents[operation][argumentsToOperands[argument]];
          llvm::transform(llvm::zip(extent.negative, offset),
                          extent.negative.begin(),
                          [](std::tuple<int64_t, int64_t> x) {
                            return std::min(std::get<0>(x), std::get<1>(x));
                          });
          llvm::transform(llvm::zip(extent.positive, offset),
                          extent.positive.begin(),
                          [](std::tuple<int64_t, int64_t> x) {
                            return std::max(std::get<0>(x), std::get<1>(x));
                          });
        }
      });
    });
  }

  const Extent *lookupExtent(Operation *op, Value value) const {
    auto operation = extents.find(op);
    if (operation == extents.end())
      return nullptr;
    auto extent = operation->second.find(value);
    if (extent == operation->second.end())
      return nullptr;
    return &extent->second;
  }

private:
  llvm::DenseMap<Operation *, llvm::DenseMap<Value, Extent>> extents;
};

struct ShapeInferencePass : public FunctionPass<ShapeInferencePass> {
  void runOnFunction() override;
};

/// Extend the loop bounds for the given use
LogicalResult extendBounds(Operation *op, OpOperand &use,
                           const AccessExtents &extents,
                           SmallVector<int64_t, 3> &lower,
                           SmallVector<int64_t, 3> &upper) {
  // Copy the bounds of store ops
  if (auto storeOp = dyn_cast<stencil::StoreOp>(use.getOwner())) {
    llvm::transform(llvm::zip(lower, storeOp.getLB()), lower.begin(),
                    [](std::tuple<int64_t, int64_t> x) {
                      return std::min(std::get<0>(x), std::get<1>(x));
                    });
    llvm::transform(llvm::zip(upper, storeOp.getUB()), upper.begin(),
                    [](std::tuple<int64_t, int64_t> x) {
                      return std::max(std::get<0>(x), std::get<1>(x));
                    });
  }
  // Extend the bounds of apply ops
  if (auto applyOp = dyn_cast<stencil::ApplyOp>(use.getOwner())) {
    auto lb = applyOp.getLB();
    auto ub = applyOp.getUB();
    // Extend loop bounds by extents
    auto opExtents = extents.lookupExtent(applyOp.getOperation(), use.get());
    if (!opExtents) {
      return op->emitError("cannot compute valid extents");
    }
    llvm::transform(llvm::zip(lb, opExtents->negative), lb.begin(),
                    [](std::tuple<int64_t, int64_t> x) {
                      return std::get<0>(x) + std::get<1>(x);
                    });
    llvm::transform(llvm::zip(ub, opExtents->positive), ub.begin(),
                    [](std::tuple<int64_t, int64_t> x) {
                      return std::get<0>(x) + std::get<1>(x);
                    });
    llvm::transform(llvm::zip(lower, lb), lower.begin(),
                    [](std::tuple<int64_t, int64_t> x) {
                      return std::min(std::get<0>(x), std::get<1>(x));
                    });
    llvm::transform(llvm::zip(upper, ub), upper.begin(),
                    [](std::tuple<int64_t, int64_t> x) {
                      return std::max(std::get<0>(x), std::get<1>(x));
                    });
  }
  return success();
}

/// Check if the range is empty
bool isEmpty(SmallVector<int64_t, 3> lower, SmallVector<int64_t, 3> upper) {
  assert(lower.size() == upper.size() &&
         "expected both vectors have equal size");
  for (size_t i = 0, e = lower.size(); i != e; ++i) {
    if (lower[i] >= upper[i])
      return true;
  }
  return false;
}

LogicalResult inferShapes(stencil::ApplyOp applyOp,
                          const AccessExtents &extents) {
  // Initial lower and upper bounds
  SmallVector<int64_t, 3> lower = {std::numeric_limits<int64_t>::max(),
                                   std::numeric_limits<int64_t>::max(),
                                   std::numeric_limits<int64_t>::max()};
  SmallVector<int64_t, 3> upper = {std::numeric_limits<int64_t>::min(),
                                   std::numeric_limits<int64_t>::min(),
                                   std::numeric_limits<int64_t>::min()};
  // Check the results of the apply op are used
  if (llvm::all_of(applyOp.getResults(),
                   [](Value result) { return result->getUses().empty(); }))
    return applyOp.emitError("failed to find use for apply op");

  // Iterate all uses and extend the bounds
  for (auto result : applyOp.getResults()) {
    for (OpOperand &use : result->getUses()) {
      if (failed(
              extendBounds(applyOp.getOperation(), use, extents, lower, upper)))
        return failure();
    }
  }
  assert(!isEmpty(lower, upper) && "failed to derive valid bounds");
  applyOp.setLB(lower);
  applyOp.setUB(upper);
  return success();
}

LogicalResult inferShapes(stencil::LoadOp loadOp,
                          const AccessExtents &extents) {
  // Initial lower and upper bounds
  SmallVector<int64_t, 3> lower = {std::numeric_limits<int64_t>::max(),
                                   std::numeric_limits<int64_t>::max(),
                                   std::numeric_limits<int64_t>::max()};
  SmallVector<int64_t, 3> upper = {std::numeric_limits<int64_t>::min(),
                                   std::numeric_limits<int64_t>::min(),
                                   std::numeric_limits<int64_t>::min()};
  // Check the result of the load op is used
  if (loadOp.getResult()->getUses().empty())
    return loadOp.emitError("failed to find use for load op");

  // Iterate all uses and extend the bounds
  for (OpOperand &use : loadOp.getResult()->getUses()) {
    if (failed(extendBounds(loadOp.getOperation(), use, extents, lower, upper)))
      return failure();
  }
  assert(!isEmpty(lower, upper) && "failed to derive valid bounds");
  loadOp.setLB(lower);
  loadOp.setUB(upper);
  return success();
}

LogicalResult assertShape(stencil::AssertOp assertOp,
                          const AccessExtents &extents) {
  // Helper lambda to check bounds
  auto verifyBounds = [&](const SmallVector<int64_t, 3> &lb,
                          const SmallVector<int64_t, 3> &ub) {
    if (llvm::any_of(llvm::zip(lb, assertOp.getLB()),
                     [](std::tuple<int64_t, int64_t> x) {
                       return std::get<0>(x) < std::get<1>(x);
                     }) ||
        llvm::any_of(llvm::zip(ub, assertOp.getUB()),
                     [](std::tuple<int64_t, int64_t> x) {
                       return std::get<0>(x) > std::get<1>(x);
                     }))
      return false;
    return true;
  };

  // Verify for every use that the access bounds fit the field
  for (OpOperand &use : assertOp.field()->getUses()) {
    if (auto storeOp = dyn_cast<stencil::StoreOp>(use.getOwner())) {
      if (!verifyBounds(storeOp.getLB(), storeOp.getUB()))
        return assertOp.emitOpError("inferred shapes too large");
    }
    if (auto loadOp = dyn_cast<stencil::LoadOp>(use.getOwner())) {
      if (!verifyBounds(loadOp.getLB(), loadOp.getUB()))
        return assertOp.emitOpError("inferred shapes too large");
    }
  }
  return success();
}

} // namespace

void ShapeInferencePass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!stencil::StencilDialect::isStencilProgram(funcOp))
    return;

  // Compute the extent analysis
  AccessExtents &extents = getAnalysis<AccessExtents>();

  // Go through the operations in reverse order
  Block &entryBlock = funcOp.getOperation()->getRegion(0).front();
  for (auto op = entryBlock.rbegin(); op != entryBlock.rend(); ++op) {
    if (auto applyOp = dyn_cast<stencil::ApplyOp>(*op)) {
      if (failed(inferShapes(applyOp, extents)))
        signalPassFailure();
    }
    if (auto loadOp = dyn_cast<stencil::LoadOp>(*op)) {
      if (failed(inferShapes(loadOp, extents)))
        signalPassFailure();
    }
    if (auto assertOp = dyn_cast<stencil::AssertOp>(*op)) {
      if (failed(assertShape(assertOp, extents)))
        signalPassFailure();
    }
  }
}

static PassRegistration<ShapeInferencePass>
    pass("stencil-shape-inference",
         "Infer the shapes of stencil loads and stencil applies.");
