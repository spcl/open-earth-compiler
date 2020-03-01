#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>

using namespace mlir;

namespace {

// Method updating the loop body of the apply op
void unrollStencilApply(stencil::ApplyOp applyOp, unsigned unrollFactor) {
  // Setup the builder and
  OpBuilder b(applyOp);
  
  // Set the insertion point before the return operation
  auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
  b.setInsertionPoint(returnOp);
  
  // Allocate container to store ther results of the unrolled iterations
  // (Store the results of the first iteration)
  std::vector<SmallVector<Value, 4>> resultBuffer;
  resultBuffer.push_back(SmallVector<Value, 4>(returnOp.getOperands()));
  // Clone the body of the apply op and replicate it multiple times
  for (unsigned i = 1, e = unrollFactor; i != e; ++i) {
    auto clonedOp = applyOp.clone();
    // Update offsets on function clone
    clonedOp.getBody()->walk([&](stencil::AccessOp accessOp) {
      SmallVector<int64_t, 3> current = accessOp.getOffset();
      ArrayAttr sum = b.getI64ArrayAttr(
          llvm::makeArrayRef({current[0], current[1] + i, current[2]}));
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
    clonedOp.erase();
  }

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
  void runOnFunction() override;
};

void StencilUnrollingPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  // Only run on functions marked as stencil programs
  if (!stencil::StencilDialect::isStencilProgram(funcOp))
    return;
  // Unroll all stencil apply ops
  funcOp.walk(
      [&](stencil::ApplyOp applyOp) { unrollStencilApply(applyOp, 2); });
}

} // namespace

std::unique_ptr<OpPassBase<FuncOp>>
mlir::stencil::createStencilUnrollingPass() {
  return std::make_unique<StencilUnrollingPass>();
}

static PassRegistration<StencilUnrollingPass> pass("stencil-unrolling",
                                                   "Unroll stencil apply ops");
