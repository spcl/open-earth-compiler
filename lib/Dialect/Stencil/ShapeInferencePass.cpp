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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include <bits/stdint-intn.h>
#include <limits>

using namespace mlir;
using namespace stencil;

namespace {

// Apply binary function element by element
SmallVector<int64_t, 3>
applyFun(const SmallVector<int64_t, 3> &x, const SmallVector<int64_t, 3> &y,
         const std::function<int64_t(int64_t, int64_t)> &fun) {
  assert(x.size() == y.size() && "expected both vectors have equal size");
  SmallVector<int64_t, 3> result;
  result.resize(x.size());
  for (int i = 0, e = x.size(); i != e; ++i) {
    result[i] = fun(x[i], y[i]);
  }
  return result;
}

/// This class computes for every stencil apply operand
/// the minimal bounding box containing all access offsets.
class AccessExtents {
  // This struct stores the positive and negative extends
  struct Extent {
    SmallVector<int64_t, 3> positive;
    SmallVector<int64_t, 3> negative;
  };

public:
  AccessExtents(Operation *op) {
    // Walk all apply ops of the stencil program
    op->walk([&](stencil::ApplyOp applyOp) {
      auto operation = applyOp.getOperation();
      // Compute mapping between operands and block arguments
      llvm::DenseMap<Value *, Value *> argumentsToOperands;
      for (int i = 0, e = applyOp.operands().size(); i != e; ++i) {
        argumentsToOperands[applyOp.getBody()->getArgument(i)] =
            applyOp.operands()[i];
        extents[operation][applyOp.operands()[i]] = {{0, 0, 0}, {0, 0, 0}};
      }
      // Walk the access ops and update the extent
      applyOp.walk([&](stencil::AccessOp accessOp) {
        auto offset = accessOp.getOffset();
        auto argument = accessOp.getOperand();
        auto &ext = extents[operation][argumentsToOperands[argument]];
        ext.negative = applyFun(ext.negative, offset, [](int64_t x, int64_t y) {
          return std::min(x, y);
        });
        ext.positive = applyFun(ext.positive, offset, [](int64_t x, int64_t y) {
          return std::max(x, y);
        });
      });
    });
  }

  const Extent *lookupExtent(Operation *op, Value *value) const {
    auto operation = extents.find(op);
    if (operation == extents.end())
      return nullptr;
    auto extent = operation->second.find(value);
    if (extent == operation->second.end())
      return nullptr;
    return &extent->second;
  }

private:
  llvm::DenseMap<Operation *, llvm::DenseMap<Value *, Extent>> extents;
};

struct ShapeInferencePass : public FunctionPass<ShapeInferencePass> {
  void runOnFunction() override;
};

/// Extend the loop bounds for the given use
void extendBounds(OpOperand &use, const AccessExtents &extents,
                  SmallVector<int64_t, 3> &lower,
                  SmallVector<int64_t, 3> &upper) {
  // Copy the bounds of store ops
  if (auto storeOp = dyn_cast<stencil::StoreOp>(use.getOwner())) {
    lower = applyFun(lower, storeOp.getLB(),
                     [](int64_t x, int64_t y) { return std::min(x, y); });
    upper = applyFun(upper, storeOp.getUB(),
                     [](int64_t x, int64_t y) { return std::max(x, y); });
  }
  // Extend the bounds of apply ops
  if (auto applyOp = dyn_cast<stencil::ApplyOp>(use.getOwner())) {
    auto lb = applyOp.getLB();
    auto ub = applyOp.getUB();
    // Extend loop bounds by extents
    auto opExtents = extents.lookupExtent(applyOp.getOperation(), use.get());
    assert(opExtents && "expected valid access extent analysis");
    lb = applyFun(lb, opExtents->negative,
                  [](int64_t x, int64_t y) { return x + y; });
    ub = applyFun(ub, opExtents->positive,
                  [](int64_t x, int64_t y) { return x + y; });
    lower = applyFun(lower, lb,
                     [](int64_t x, int64_t y) { return std::min(x, y); });
    upper = applyFun(upper, ub,
                     [](int64_t x, int64_t y) { return std::max(x, y); });
  }
}

/// Check if the range is empty
bool isEmpty(SmallVector<int64_t, 3> lower, SmallVector<int64_t, 3> upper) {
  assert(lower.size() == upper.size() &&
         "expected both vectors have equal size");
  for (int i = 0, e = lower.size(); i != e; ++i) {
    if (lower[i] >= upper[i])
      return true;
  }
  return false;
}

bool inferShapes(stencil::ApplyOp applyOp, const AccessExtents &extents) {
  // Initial lower and upper bounds
  SmallVector<int64_t, 3> lower = {std::numeric_limits<int64_t>::max(),
                                   std::numeric_limits<int64_t>::max(),
                                   std::numeric_limits<int64_t>::max()};
  SmallVector<int64_t, 3> upper = {std::numeric_limits<int64_t>::min(),
                                   std::numeric_limits<int64_t>::min(),
                                   std::numeric_limits<int64_t>::min()};
  // Iterate all uses and extend the bounds
  for (auto result : applyOp.getResults()) {
    for (OpOperand &use : result->getUses()) {
      extendBounds(use, extents, lower, upper);
    }
  }
  if (isEmpty(lower, upper)) {
    applyOp.emitError("failed to derive non empty bounds");
    return false;
  }
  applyOp.setLB(lower);
  applyOp.setUB(upper);
  return true;
}

bool inferShapes(stencil::LoadOp loadOp, const AccessExtents &extents) {
  // Initial lower and upper bounds
  SmallVector<int64_t, 3> lower = {std::numeric_limits<int64_t>::max(),
                                   std::numeric_limits<int64_t>::max(),
                                   std::numeric_limits<int64_t>::max()};
  SmallVector<int64_t, 3> upper = {std::numeric_limits<int64_t>::min(),
                                   std::numeric_limits<int64_t>::min(),
                                   std::numeric_limits<int64_t>::min()};
  // Iterate all uses and extend the bounds
  for (OpOperand &use : loadOp.getResult()->getUses()) {
    extendBounds(use, extents, lower, upper);
  }
  if (isEmpty(lower, upper)) {
    loadOp.emitError("failed to derive non empty bounds");
    return false;
  }
  loadOp.setLB(lower);
  loadOp.setUB(upper);
  return true;
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
      if (!inferShapes(applyOp, extents))
        signalPassFailure();
    }
    if (auto loadOp = dyn_cast<stencil::LoadOp>(*op)) {
      if (!inferShapes(loadOp, extents))
        signalPassFailure();
    }
  }
}

static PassRegistration<ShapeInferencePass>
    pass("stencil-shape-inference", "Infer the shapes of views and fields.");
