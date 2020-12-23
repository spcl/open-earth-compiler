#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>

using namespace mlir;
using namespace stencil;

namespace {

struct StencilUnrollingPass
    : public StencilUnrollingPassBase<StencilUnrollingPass> {

  void runOnFunction() override;

protected:
  void unrollStencilApply(stencil::ApplyOp applyOp);
  void addPeelIteration(stencil::ApplyOp applyOp);

  void makePeelIteration(stencil::ReturnOp returnOp, unsigned tripCount);
  stencil::ReturnOp cloneBody(stencil::ApplyOp from, stencil::ApplyOp to,
                              OpBuilder &builder);
};

stencil::ReturnOp StencilUnrollingPass::cloneBody(stencil::ApplyOp from,
                                                  stencil::ApplyOp to,
                                                  OpBuilder &builder) {
  // Setup the argument mapper
  BlockAndValueMapping mapper;
  for (auto it : llvm::zip(from.getBody()->getArguments(),
                           to.getBody()->getArguments())) {
    mapper.map(std::get<0>(it), std::get<1>(it));
  }
  // Clone the apply op body
  Operation *last = nullptr;
  for (auto &op : from.getBody()->getOperations()) {
    last = builder.clone(op, mapper);
  }
  return cast<stencil::ReturnOp>(last);
}

void StencilUnrollingPass::unrollStencilApply(stencil::ApplyOp applyOp) {
  // Setup the builder and
  OpBuilder b(applyOp);

  // Prepare a clone containing a single iteration and an argument mapper
  auto clonedOp = applyOp.clone();

  // Keep a list of the return ops for all unrolled loop iterations
  SmallVector<stencil::ReturnOp, 4> loopIterations = {
      cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator())};

  // Keep unrolling until there is one returnOp for every iteration
  b.setInsertionPointToEnd(applyOp.getBody());
  while (loopIterations.size() < unrollFactor) {
    // Update the offsets of the clone
    clonedOp.getBody()->walk([&](stencil::ShiftOp shiftOp) {
      Index offset(kIndexSize, 0);
      offset[unrollIndex] = 1;
      shiftOp.shiftByOffset(offset);
    });
    // Clone the body and store the return op
    loopIterations.push_back(cloneBody(clonedOp, applyOp, b));
  }
  clonedOp.erase();

  // Compute the results for the unrolled apply op
  SmallVector<Value, 16> newResults;
  for (unsigned i = 0, e = applyOp.getNumResults(); i != e; ++i) {
    llvm::transform(
        loopIterations, std::back_inserter(newResults),
        [&](stencil::ReturnOp returnOp) { return returnOp.getOperand(i); });
  }
  for (auto returnOp : loopIterations) {
    returnOp.erase();
  }

  // Create a new return op returning all results
  SmallVector<int64_t, kIndexSize> unroll(kIndexSize, 1);
  unroll[unrollIndex] = unrollFactor;
  b.create<stencil::ReturnOp>(loopIterations.front().getLoc(), newResults,
                              b.getI64ArrayAttr(unroll));
}

void StencilUnrollingPass::makePeelIteration(stencil::ReturnOp returnOp,
                                             unsigned tripCount) {
  // Create empty store for all iterations that exceed the trip count
  SmallVector<Value, 16> newOperands;
  for (auto en : llvm::enumerate(returnOp.getOperands())) {
    if (en.index() % unrollFactor >= tripCount) {
      en.value().getDefiningOp()->setOperands({});
    }
  }
}

void StencilUnrollingPass::addPeelIteration(stencil::ApplyOp applyOp) {
  // Check if the domain size is not a multiple of the unroll factor
  auto shapeOp = cast<ShapeOp>(applyOp.getOperation());
  auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
  auto domainSize = shapeOp.getUB()[unrollIndex] - shapeOp.getLB()[unrollIndex];
  if (domainSize % unrollFactor != 0) {
    if (domainSize < unrollFactor) {
      makePeelIteration(returnOp, domainSize);
    } else {
      // Setup the builder
      OpBuilder b(applyOp);
      auto loc = applyOp.getLoc();

      // Introduce a second apply to handle the peel domain
      auto peelOp = cast<stencil::ApplyOp>(b.clone(*applyOp.getOperation()));
      auto bodyOp = cast<stencil::ApplyOp>(b.clone(*applyOp.getOperation()));

      // // Adapt the shape of the two apply ops
      auto lb = shapeOp.getLB();
      auto ub = shapeOp.getUB();
      int64_t split = ub[unrollIndex] - domainSize % unrollFactor;
      lb[unrollIndex] = split;
      ub[unrollIndex] = split;
      cast<ShapeOp>(peelOp.getOperation()).updateShape(lb, shapeOp.getUB());
      cast<ShapeOp>(bodyOp.getOperation()).updateShape(shapeOp.getLB(), ub);

      // Remove stores that exceed the domain size
      makePeelIteration(
          cast<stencil::ReturnOp>(peelOp.getBody()->getTerminator()),
          domainSize % unrollFactor);

      // Introduce a stencil combine to replace the uses of the original apply
      auto combineOp = b.create<stencil::CombineOp>(
          loc, applyOp.getResultTypes(), unrollIndex, split,
          bodyOp.getResults(), peelOp.getResults(), ValueRange(), ValueRange(),
          applyOp.lbAttr(), applyOp.ubAttr());

      applyOp.replaceAllUsesWith(combineOp.getResults());
      applyOp.erase();
    }
  }
}

void StencilUnrollingPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  // Only run on functions marked as stencil programs
  if (!StencilDialect::isStencilProgram(funcOp))
    return;

  // Check for valid unrolling indexes
  if (unrollIndex == 0) {
    funcOp.emitError("unrolling the innermost loop is not supported");
    signalPassFailure();
    return;
  }

  // Check shape inference has been executed
  auto result = funcOp->walk([&](stencil::ShapeOp shapeOp) {
    if (!shapeOp.hasShape())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    funcOp.emitOpError("execute shape inference before stencil unrolling");
    signalPassFailure();
    return;
  }

  // Collect the stencil apply operations
  SmallVector<stencil::ApplyOp, 16> workList;
  funcOp.walk([&](stencil::ApplyOp applyOp) { workList.push_back(applyOp); });

  // Unroll the stencil apply operations
  for (auto applyOp : workList) {
    unrollStencilApply(applyOp);
    addPeelIteration(applyOp);
  }
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createStencilUnrollingPass() {
  return std::make_unique<StencilUnrollingPass>();
}
