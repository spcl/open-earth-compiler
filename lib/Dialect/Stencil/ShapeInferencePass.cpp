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
        auto unrollDim = returnOp.getUnrollDim();
        auto unrollFac = returnOp.getUnrollFac();
        // Limit the unroll factor to the loop length if available
        auto shapeOp = cast<ShapeOp>(applyOp.getOperation());
        if (shapeOp.hasShape()) {
          unrollFac = min(unrollFac, shapeOp.getUB()[unrollDim] -
                                         shapeOp.getLB()[unrollDim]);
        }
        for (auto operand : applyOp.getOperands()) {
          if (extents[operation].count(operand) == 1) {
            auto &positive = extents[operation][operand].positive;
            positive[unrollDim] -= unrollFac - 1;
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

protected:
  // Shape extension
  void updateStorageShape(ShapeOp shapeOp, ArrayRef<int64_t> lb,
                          ArrayRef<std::int64_t> ub);
  bool isShapeExtension(ShapeOp current, ArrayRef<int64_t> lb,
                        ArrayRef<std::int64_t> ub);
  std::tuple<Index, Index> getResultShape(Operation *definingOp, Value result);
  LogicalResult extendStorageShape(ShapeOp shapeOp, Value temp);

  // Shape inference
  void updateShape(ShapeOp shapeOp, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub);
  LogicalResult updateBounds(OpOperand &use, const AccessExtents &extents,
                             Index &lower, Index &upper);
  LogicalResult inferShapes(ShapeOp shapeOp, const AccessExtents &extents);
};

/// Update the shape given updated bounds
void ShapeInferencePass::updateShape(ShapeOp shapeOp, ArrayRef<int64_t> lb,
                                     ArrayRef<int64_t> ub) {
  shapeOp.updateShape(lb, ub);
  // Update the region arguments of dependent shape operations
  // (needed for operations such as the stencil apply op)
  for (auto user : shapeOp.getOperation()->getUsers()) {
    if (auto shapeOp = dyn_cast<ShapeOp>(user))
      shapeOp.updateArgumentTypes();
  }
}

/// Extend the loop bounds for the given use
LogicalResult ShapeInferencePass::updateBounds(OpOperand &use,
                                               const AccessExtents &extents,
                                               Index &lower, Index &upper) {
  // Copy the bounds of store ops
  if (auto shapeOp = dyn_cast<ShapeOp>(use.getOwner())) {
    if (shapeOp.hasShape()) {
      auto lb = shapeOp.getLB();
      auto ub = shapeOp.getUB();
      // Adjust the bounds by the access extents if available
      if (auto opExtents = extents.lookupExtent(use.getOwner(), use.get())) {
        lb = applyFunElementWise(lb, opExtents->negative, std::plus<int64_t>());
        ub = applyFunElementWise(ub, opExtents->positive, std::plus<int64_t>());
      }
      // Adjust the bounds if the shape is split into subdomains
      if (auto combineOp = dyn_cast<stencil::CombineOp>(use.getOwner())) {
        if (combineOp.isLowerOperand(use.getOperandNumber()))
          ub[combineOp.dim()] = min(combineOp.getIndex(), ub[combineOp.dim()]);
        if (combineOp.isUpperOperand(use.getOperandNumber()))
          lb[combineOp.dim()] = max(combineOp.getIndex(), lb[combineOp.dim()]);
      }

      // Verify the shape is not empty
      if (llvm::any_of(llvm::zip(lb, ub), [](std::tuple<int64_t, int64_t> x) {
            return std::get<0>(x) >= std::get<1>(x);
          }))
        return success();

      // Update the bounds given the extend of the use
      lower = lower.empty() ? lb : lower;
      upper = upper.empty() ? ub : upper;
      if (lower.size() != shapeOp.getRank() ||
          upper.size() != shapeOp.getRank())
        return shapeOp.emitOpError("expected operations to have the same rank");
      lower = applyFunElementWise(lower, lb, min);
      upper = applyFunElementWise(upper, ub, max);
    }
  }
  return success();
}

/// Infer the shape as the maximum bounding box the consumer shapes
LogicalResult ShapeInferencePass::inferShapes(ShapeOp shapeOp,
                                              const AccessExtents &extents) {
  Index lb, ub;
  // Iterate over all uses and adjust the bounds
  for (auto result : shapeOp.getOperation()->getResults()) {
    for (OpOperand &use : result.getUses()) {
      if (failed(updateBounds(use, extents, lb, ub)))
        return failure();
    }
  }
  // Update the shape if the bounds are known
  if (lb.size() != 0 && ub.size() != 0)
    updateShape(shapeOp, lb, ub);
  return success();
}

/// Compute the shape for a given result of a defining op
std::tuple<Index, Index>
ShapeInferencePass::getResultShape(Operation *definingOp, Value result) {
  // If the defining op is an apply op return its shape
  if (auto applyOp = dyn_cast<stencil::ApplyOp>(definingOp)) {
    auto shapeOp = cast<ShapeOp>(applyOp.getOperation());
    if (shapeOp.hasShape()) {
      return std::make_tuple(shapeOp.getLB(), shapeOp.getUB());
    } else {
      return std::make_tuple(
          Index(kIndexSize, std::numeric_limits<int64_t>::max()),
          Index(kIndexSize, std::numeric_limits<int64_t>::min()));
    }
  }
  // If the defining op is a combine check if it is an extra parameter
  if (auto combineOp = dyn_cast<stencil::CombineOp>(definingOp)) {
    auto it = llvm::find(combineOp.getResults(), result);
    assert(it != combineOp.getResults().end() &&
           "expected to find the result value");
    unsigned resultNumber = std::distance(combineOp.getResults().begin(), it);
    // If the result is not an extra result combine the shape recursively
    if (auto num = combineOp.getLowerOperandNumber(resultNumber)) {
      Index lb1, lb2, ub1, ub2;
      std::tie(lb1, ub1) =
          getResultShape(combineOp.lower()[num.getValue()].getDefiningOp(),
                         combineOp.lower()[num.getValue()]);
      std::tie(lb2, ub2) =
          getResultShape(combineOp.upper()[num.getValue()].getDefiningOp(),
                         combineOp.upper()[num.getValue()]);
      return std::make_tuple(applyFunElementWise(lb1, lb2, min),
                             applyFunElementWise(ub1, ub2, max));
    }
    // If the result is a lower extra result compute the shape recursively
    if (auto num = combineOp.getLowerExtraOperandNumber(resultNumber)) {
      return getResultShape(
          combineOp.lowerext()[num.getValue()].getDefiningOp(),
          combineOp.lowerext()[num.getValue()]);
    }
    // If the result is an upper extra result compute the shape recursively
    if (auto num = combineOp.getUpperExtraOperandNumber(resultNumber)) {
      return getResultShape(
          combineOp.upperext()[num.getValue()].getDefiningOp(),
          combineOp.upperext()[num.getValue()]);
    }
  }
  llvm_unreachable("expected an apply or a combine op");
  return {};
}

/// Update the shape given the old and the new shape op
void ShapeInferencePass::updateStorageShape(ShapeOp shapeOp,
                                            ArrayRef<int64_t> lb,
                                            ArrayRef<std::int64_t> ub) {
  // Compute the update bounds
  auto lower = applyFunElementWise(shapeOp.getLB(), lb, min);
  auto upper = applyFunElementWise(shapeOp.getUB(), ub, max);
  updateShape(shapeOp, lower, upper);
}

/// Check it is a valid shape extension
bool ShapeInferencePass::isShapeExtension(ShapeOp current, ArrayRef<int64_t> lb,
                                          ArrayRef<std::int64_t> ub) {
  if (llvm::all_of(llvm::zip(lb, current.getLB()),
                   [](std::tuple<int64_t, int64_t> x) {
                     return std::get<0>(x) <= std::get<1>(x);
                   }) &&
      llvm::all_of(llvm::zip(ub, current.getUB()),
                   [](std::tuple<int64_t, int64_t> x) {
                     return std::get<0>(x) >= std::get<1>(x);
                   }))
    return true;
  return false;
}

/// Update the storage shape if needed
LogicalResult ShapeInferencePass::extendStorageShape(ShapeOp shapeOp,
                                                     Value temp) {
  Index lb, ub;
  std::tie(lb, ub) = getResultShape(temp.getDefiningOp(), temp);
  // Update the shape
  if (shapeOp.getLB() != lb || shapeOp.getUB() != ub) {
    // Update the storage if it is an extension
    if (isShapeExtension(shapeOp, lb, ub)) {
      updateStorageShape(shapeOp, lb, ub);
      return success();
    }
    // Emit an error if the update shrinks the shape
    shapeOp.emitOpError("cannot shrink the storage size");
    signalPassFailure();
  }
  return failure();
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

  // Extend the shape of stores if the flag is set
  if (extendStorage) {
    // Update the store shapes and issue a warning
    funcOp.walk([&](stencil::StoreOp storeOp) {
      if (succeeded(extendStorageShape(storeOp.getOperation(), storeOp.temp())))
        storeOp.emitWarning(
            "adapted shape to match the write set of the defining op");
    });
    // Update the buffer shapes
    funcOp.walk([&](stencil::BufferOp bufferOp) {
      extendStorageShape(bufferOp.getOperation(), bufferOp.temp());
    });
  }
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
