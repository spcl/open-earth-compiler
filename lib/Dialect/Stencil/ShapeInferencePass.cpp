#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

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
      // Walk the access operations and update the extent
      applyOp.walk([&](ExtentOp extentOp) {
        Index lb, ub;
        std::tie(lb, ub) = extentOp.getAccessExtent();
        auto temp = extentOp.getTemp();
        if (extents[operation].count(argToOperand[temp]) == 0) {
          // Initialize the extents with the current offset
          extents[operation][argToOperand[temp]].negative = lb;
          extents[operation][argToOperand[temp]].positive = ub;
        } else {
          // Extend the extents with the current offset
          auto &negative = extents[operation][argToOperand[temp]].negative;
          auto &positive = extents[operation][argToOperand[temp]].positive;
          negative = applyFunElementWise(negative, lb, min);
          positive = applyFunElementWise(positive, ub, max);
        }
      });
      // Subtract the unroll factor minus one from the positive extent
      auto returnOp =
          cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
      if (returnOp.unroll().hasValue()) {
        for (auto operand : applyOp.getOperands()) {
          if (extents[operation].count(operand) == 1) {
            auto &positive = extents[operation][operand].positive;
            positive = applyFunElementWise(
                positive, returnOp.getUnroll(),
                [](int64_t x, int64_t y) { return x - y + 1; });
          }
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
LogicalResult adjustBounds(const OpOperand &use, const AccessExtents &extents,
                           Index &lower, Index &upper) {
  // Copy the bounds of store ops
  if (auto shapeOp = dyn_cast<ShapeOp>(use.getOwner())) {
    auto lb = shapeOp.getLB();
    auto ub = shapeOp.getUB();
    // Adjust the bounds by the access extents if available
    if (auto opExtents = extents.lookupExtent(use.getOwner(), use.get())) {
      lb = applyFunElementWise(lb, opExtents->negative, std::plus<int64_t>());
      ub = applyFunElementWise(ub, opExtents->positive, std::plus<int64_t>());
    }
    // Adjust the bounds if the shape is split into subdomains
    if (auto combineOp = dyn_cast<stencil::CombineOp>(use.getOwner())) {
      if (llvm::is_contained(combineOp.lower(), use.get()) ||
          llvm::is_contained(combineOp.lowerext(), use.get()))
        ub[combineOp.dim()] = combineOp.index();
      if (llvm::is_contained(combineOp.upper(), use.get()) ||
          llvm::is_contained(combineOp.upperext(), use.get()))
        lb[combineOp.dim()] = combineOp.index();
    }
    // Update the lower and upper bounds
    if (lower.empty() && upper.empty()) {
      lower = lb;
      upper = ub;
    } else {
      if (lower.size() != shapeOp.getRank() ||
          upper.size() != shapeOp.getRank())
        return shapeOp.emitOpError("expected operations to have the same rank");
      lower = applyFunElementWise(lower, lb, min);
      upper = applyFunElementWise(upper, ub, max);
    }
  }
  return success();
}

LogicalResult inferShapes(ShapeOp shapeOp, const AccessExtents &extents) {
  Index lb, ub;
  // Iterate over all uses and adjust the bounds
  for (auto result : shapeOp.getOperation()->getResults()) {
    for (OpOperand &use : result.getUses()) {
      if (failed(adjustBounds(use, extents, lb, ub)))
        return failure();
    }
  }
  // Update the the operation bounds
  auto shape = applyFunElementWise(ub, lb, std::minus<int64_t>());
  if (shape.empty())
    return shapeOp.emitOpError("expected shape to have non-zero size");
  if (llvm::any_of(shape, [](int64_t size) { return size < 1; }))
    return shapeOp.emitOpError("expected shape to have non-zero entries");
  shapeOp.updateShape(lb, ub);

  // Update the region arguments of dependent shape operations
  // (needed for operations such as the stencil apply op)
  for (auto user : shapeOp.getOperation()->getUsers()) {
    if (auto shapeOp = dyn_cast<ShapeOp>(user))
      shapeOp.updateArgumentTypes();
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
    if (auto shapeOp = dyn_cast<ShapeOp>(*op)) {
      // Clear the inferred shapes
      shapeOp.clearInferredShape();
      // Compute the shape
      if (!shapeOp.hasShape()) {
        if (failed(inferShapes(shapeOp, extents))) {
          signalPassFailure();
          return;
        }
      }
    }
  }

  // Extend the shape of stencil stores if the loop executes on a bigger domain
  // (TODO instead store outputs only on a smaller domain)
  funcOp.walk([](stencil::StoreOp storeOp) {
    auto applyOp = cast<ShapeOp>(storeOp.temp().getDefiningOp());
    auto shapeOp = cast<ShapeOp>(storeOp.getOperation());
    shapeOp.updateShape(applyOp.getLB(), applyOp.getUB());
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
