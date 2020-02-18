#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTraits.h"
#include "mlir/Dialect/StandardOps/Ops.h"
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
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>

using namespace mlir;

namespace {

// Helper method to clone an operation
Operation *cloneOperation(OpBuilder &builder, Operation *op) {
  auto clonedOp = builder.clone(*op);
  for (unsigned i = 0, e = clonedOp->getNumResults(); i != e; ++i) {
    op->getResult(i).replaceAllUsesWith(clonedOp->getResult(i));
  }
  op->erase();
  return clonedOp;
}

// Helper method picking the next operation to schedule
Operation *getNextReadyOp(const std::vector<Operation *> &remainingOps,
                          const llvm::DenseSet<Value> &lifeValues) {
  // Select remaining op if all parameters are life
  for (auto remainingOp : remainingOps) {
    if (llvm::all_of(remainingOp->getOperands(), [&](Value value) {
          return lifeValues.count(value) != 0;
        })) {
      return remainingOp;
    }
  }
  return nullptr;
}

void updateLifeValuesAndScheduledOps(llvm::DenseSet<Value> &lifeValues,
                                     llvm::DenseSet<Operation *> &scheduledOps,
                                     Operation *clonedOp) {
  // Add all results to life values
  for (auto result : clonedOp->getResults()) {
    lifeValues.insert(result);
  }
  scheduledOps.insert(clonedOp);

  // Update life values
  for (auto value : clonedOp->getOperands()) {
    if (llvm::all_of(value.getUsers(), [&](Operation *user) {
          return scheduledOps.count(user) != 0;
        }))
      lifeValues.erase(value);
  }
}

struct StencilSchedulePass
    : public OperationPass<StencilSchedulePass, stencil::ApplyOp> {
  void runOnOperation() override;
};

void StencilSchedulePass::runOnOperation() {
  auto applyOp = getOperation();

  // Walk all operations and store constants and accesses separately
  std::vector<Operation *> accessOps;
  std::vector<Operation *> constantOps;
  std::vector<Operation *> arithmeticOps;
  applyOp.getBody()->walk([&](Operation *op) {
    // Store all other ops
    if (isa<stencil::AccessOp>(op)) {
      accessOps.push_back(op);
      return;
    }
    if (isa<ConstantOp>(op)) {
      constantOps.push_back(op);
      return;
    }
    arithmeticOps.push_back(op);
  });
  std::reverse(accessOps.begin(), accessOps.end());

  // Clone operation by operation
  OpBuilder builder(applyOp.getBody());

  // Keep scheduling operations
  llvm::DenseSet<Value> lifeValues;
  llvm::DenseSet<Operation *> scheduledOps;
  std::map<size_t, unsigned> lifeFrequencies;
  // Schedule all constants
  for (auto constantOp : constantOps) {
    auto clonedOp = cloneOperation(builder, constantOp);
    updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
    lifeFrequencies[lifeValues.size()]++;
  }
  // Schedule accesses and arithmetic Ops
  while (!(accessOps.empty() && arithmeticOps.empty())) {
    // Schedule all arithmetic Ops that are ready for execution
    while (Operation *next = getNextReadyOp(arithmeticOps, lifeValues)) {
      auto clonedOp = cloneOperation(builder, next);
      updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
      arithmeticOps.erase(llvm::find(arithmeticOps, next));
      lifeFrequencies[lifeValues.size()]++;
    }

    // Schedule the next arithmetic op
    if (!accessOps.empty()) {
      auto clonedOp = cloneOperation(builder, accessOps.back());
      updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
      accessOps.pop_back();
      lifeFrequencies[lifeValues.size()]++;
    }
  }

  // Print the top5 life value frequencies
  llvm::outs() << "// Life Value Frequencies\n"
               << "// ======================\n";
  for(auto it = lifeFrequencies.begin(); it != lifeFrequencies.end(); ++it)
    llvm::outs() << "// - " << it->first << "(" << it->second << ")\n";
}

} // namespace

std::unique_ptr<OpPassBase<stencil::ApplyOp>>
stencil::createStencilSchedulePass() {
  return std::make_unique<StencilSchedulePass>();
}

void stencil::createStencilSchedulePipeline(OpPassManager &pm) {
  auto &funcPm = pm.nest<FuncOp>();
  //funcPm.addPass(createStencilPreShufflePass());
  funcPm.addPass(createStencilShufflePass());
  //funcPm.addPass(createStencilPostShufflePass());
  funcPm.addPass(createStencilSchedulePass());
}

void stencil::createStencilScheduleOnlyPipeline(OpPassManager &pm) {
  auto &funcPm = pm.nest<FuncOp>();
  funcPm.addPass(createStencilSchedulePass());
}

// Life Value Frequencies
// ======================

// - 10(5)
// - 11(6)
// - 12(5)
// - 13(6)
// - 14(7)
// - 15(3)
// - 16(5)
// - 17(8)
// - 18(16)
// - 19(6)
// - 20(4)
// - 21(1)