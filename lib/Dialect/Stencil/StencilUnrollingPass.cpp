#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>

using namespace mlir;

namespace {

// Helper method to check if lower bound is zero
// (TODO factor this out to a stencil utils header)
bool isZero(ArrayRef<int64_t> offset) {
  return llvm::all_of(offset, [](int64_t x) {
    return x == 0 || x == stencil::kIgnoreDimension;
  });
}

// Method updating the loop body of the apply op
void unrollStencilApply(stencil::ApplyOp applyOp, unsigned unrollFactor,
                        unsigned unrollIndex) {
  // Setup the builder and
  OpBuilder b(applyOp);

  // Set the insertion point before the return operation
  auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
  b.setInsertionPoint(returnOp);

  // Allocate container to store the results of the unrolled iterations
  // (Store the results of the first iteration)
  std::vector<SmallVector<Value, 4>> resultBuffer;
  resultBuffer.push_back(SmallVector<Value, 4>(returnOp.getOperands()));
  // Clone the body of the apply op and replicate it multiple times
  auto clonedOp = applyOp.clone();
  for (unsigned i = 1, e = unrollFactor; i != e; ++i) {
    // Update offsets on function clone
    clonedOp.getBody()->walk([&](stencil::AccessOp accessOp) {
      SmallVector<int64_t, 3> current = accessOp.getOffset();
      current[unrollIndex]++;
      ArrayAttr sum = b.getI64ArrayAttr(current);
      accessOp.setAttr(accessOp.getOffsetAttrName(), sum);
    });
    // Clone the body except of the shifted apply op
    BlockAndValueMapping mapper;
    for (size_t i = 0, e = clonedOp.getBody()->getNumArguments(); i < e; ++i) {
      mapper.map(clonedOp.getBody()->getArgument(i),
                 applyOp.getBody()->getArgument(i));
    }
    for (auto &op : clonedOp.getBody()->getOperations()) {
      auto currentOp = b.clone(op, mapper);
      // Store the results after cloning the return op
      if (auto returnOp = dyn_cast<stencil::ReturnOp>(currentOp)) {
        resultBuffer.push_back(SmallVector<Value, 3>(returnOp.operands()));
        currentOp->erase();
      }
    }
  }

  clonedOp.erase();

  // Create a new return op returning all results
  SmallVector<Value, 16> newResults;
  for (unsigned i = 0, e = returnOp.getNumOperands(); i != e; ++i) {
    for (unsigned j = 0; j != unrollFactor; ++j) {
      newResults.push_back(resultBuffer[j][i]);
    }
  }
  b.create<stencil::ReturnOp>(returnOp.getLoc(), newResults,
                              stencil::convertSmallVectorToArrayAttr(
                                  {1, unrollFactor, 1}, b.getContext()));

  // Erase the original return op
  returnOp.erase();
}

struct StencilUnrollingPass : public FunctionPass<StencilUnrollingPass> {
  StencilUnrollingPass() {
    unrollFactor = 2;
    unrollIndex = 1;
  }
  StencilUnrollingPass(const StencilUnrollingPass &) {}

  void runOnFunction() override;

  Option<unsigned> unrollFactor{
      *this, "unroll-factor",
      llvm::cl::desc("number of unrolled loop iterations")};
  Option<unsigned> unrollIndex{
      *this, "unroll-index",
      llvm::cl::desc("unroll index specifying the unrolling dimension")};
};

void StencilUnrollingPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  // Only run on functions marked as stencil programs
  if (!stencil::StencilDialect::isStencilProgram(funcOp))
    return;

  // Check for valid unrolling indexes
  if (!(unrollIndex == 1 || unrollIndex == 2)) {
    funcOp.emitError("unrolling is only supported in the j-dimension (=1) or "
                     "k-dimension (=2)");
    signalPassFailure();
    return;
  }

  // Unroll all stencil apply ops
  funcOp.walk([&](stencil::ApplyOp applyOp) {
    // Check the loop bounds are known and valid
    if (!(applyOp.lb().hasValue() && applyOp.ub().hasValue())) {
      applyOp.emitError("run the shape inference passes first");
      signalPassFailure();
      return;
    }
    if (!isZero(applyOp.getLB())) {
      applyOp.emitError("run the shape shift passes first");
      signalPassFailure();
      return;
    }

    // Check the unroll factor is a multiple of the domain size
    auto ub = applyOp.getUB();
    if (ub[unrollIndex.getValue()] % unrollFactor.getValue() != 0) {
      applyOp.emitError(
          "loop bounds have to be a multiple of the unroll factor");
      signalPassFailure();
      return;
    }

    // Unroll the stencil
    unrollStencilApply(applyOp, unrollFactor.getValue(),
                       unrollIndex.getValue());
  });
}

} // namespace
std::unique_ptr<OpPassBase<FuncOp>>
mlir::stencil::createStencilUnrollingPass() {
  return std::make_unique<StencilUnrollingPass>();
}

static PassRegistration<StencilUnrollingPass> pass("stencil-unrolling",
                                                   "Unroll stencil apply ops");
