#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include <bits/stdint-intn.h>
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
    Index negative;
    Index positive;
  };

public:
  AccessExtents(Operation *op) {
    // Walk all apply ops of the stencil program
    op->walk([&](stencil::ApplyOp applyOp) {
      auto operation = applyOp.getOperation();
      // Compute mapping between operands and block arguments
      llvm::DenseMap<Value, Value> argToOperand;
      for (size_t i = 0, e = applyOp.operands().size(); i != e; ++i) {
        argToOperand[applyOp.getBody()->getArgument(i)] = applyOp.operands()[i];
      }
      // Walk the access ops and update the extent
      applyOp.walk([&](stencil::AccessOp accessOp) {
        auto offset = accessOp.getOffset();
        auto argument = accessOp.getOperand();
        if (extents[operation].count(argToOperand[argument]) == 0) {
          // Initialize the extents with the current offset
          extents[operation][argToOperand[argument]].negative = offset;
          extents[operation][argToOperand[argument]].positive = offset;
        } else {
          // Extend the extents with the current offset
          auto &negative = extents[operation][argToOperand[argument]].negative;
          auto &positive = extents[operation][argToOperand[argument]].positive;
          negative = applyFunElementWise(negative, offset, min);
          positive = applyFunElementWise(positive, offset, max);
        }
      });
      // Subtract the unroll factor minus one from the positive extent
      // (TODO shall we run shape inference before unrolling / inlining)
      auto returnOp =
          cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
      if (returnOp.unroll().hasValue()) {
        for (size_t i = 0, e = applyOp.operands().size(); i != e; ++i) {
          auto &positive = extents[operation][applyOp.getOperand(i)].positive;
          positive = applyFunElementWise(
              positive, returnOp.getUnroll(),
              [](int64_t x, int64_t y) { return x - y + 1; });
        }
      }
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

struct ShapeInferencePass : public ShapeInferencePassBase<ShapeInferencePass> {
  void runOnFunction() override;
};

/// Extend the loop bounds for the given use
LogicalResult extendBounds(Operation *op, const OpOperand &use,
                           const AccessExtents &extents, Index &lower,
                           Index &upper) {
  // Copy the bounds of store ops
  if (auto shapeOp = dyn_cast<ShapeAccess>(use.getOwner())) {
    auto lb = shapeOp.getLB();
    auto ub = shapeOp.getUB();
    // Extend the operation bounds if extent info exists
    if (auto opExtents = extents.lookupExtent(use.getOwner(), use.get())) {
      lb = applyFunElementWise(lb, opExtents->negative, std::plus<int64_t>());
      ub = applyFunElementWise(ub, opExtents->positive, std::plus<int64_t>());
    }
    // Update the lower and upper bounds
    if (lower.empty() && upper.empty()) {
      lower = lb;
      upper = ub;
    } else {
      if (lower.size() != shapeOp.getRank() ||
          upper.size() != shapeOp.getRank())
        return shapeOp.emitOpError(
            "expected all operations to have the same rank");
      lower = applyFunElementWise(lower, lb, min);
      upper = applyFunElementWise(upper, ub, max);
    }
  }
  return success();
}

LogicalResult inferShapes(ShapeInference shapeOp,
                          const AccessExtents &extents) {
  Index lb, ub;
  // Iterate all uses and extend the bounds
  for (auto result : shapeOp.getOperation()->getResults()) {
    for (OpOperand &use : result.getUses()) {
      if (failed(extendBounds(shapeOp.getOperation(), use, extents, lb, ub)))
        return failure();

      // TODO update the shape types
    }
  }
  // Update the bounds
  if (lb.empty() && ub.empty())
    return shapeOp.emitOpError("failed to compute valid shape");
  shapeOp.setOpShape(lb, ub);
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
    if (auto shapeOp = dyn_cast<ShapeInference>(*op)) {
      if (failed(inferShapes(shapeOp, extents))) {
        signalPassFailure();
        return;
      }
    }
  }
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}