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
  return clonedOp;
}

// Helper method picking the next operation to schedule
Operation *pickReadyOp(const std::vector<Operation *> &remainingOps,
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
                                     llvm::DenseSet<Operation *> &originalOps,
                                     llvm::DenseSet<Operation *> &scheduledOps,
                                     Operation *clonedOp) {
  // Add all results to life values
  for (auto result : clonedOp->getResults()) {
    lifeValues.insert(result);
  }
  scheduledOps.insert(clonedOp);

  // // Update life values
  // for (auto value : clonedOp->getOperands()) {
  //   if (llvm::all_of(value.getUsers(), [&](Operation *user) {
  //         return scheduledOps.count(user) != 0 || originalOps.count(user) !=
  //         0;
  //       }))
  //     lifeValues.erase(value);
  // }
}

struct StencilSchedulePass
    : public OperationPass<StencilSchedulePass, stencil::ApplyOp> {
  void runOnOperation() override;
};

void StencilSchedulePass::runOnOperation() {
  auto applyOp = getOperation();

  // Walk all operations and store constants and accesses separately
  std::vector<stencil::AccessOp> accessOps;
  std::vector<ConstantOp> constantOps;
  std::vector<Operation *> remainingOps;

  llvm::DenseSet<Operation *> originalOps;

  applyOp.walk([&](Operation *op) {
    if (isa<stencil::ApplyOp>(op) || isa<stencil::ReturnOp>(op)) {
      return;
    }
    if (auto accessOp = dyn_cast<stencil::AccessOp>(op)) {
      accessOps.push_back(accessOp);
      originalOps.insert(op);
      return;
    }
    if (auto constantOp = dyn_cast<ConstantOp>(op)) {
      constantOps.push_back(constantOp);
      originalOps.insert(op);
      return;
    }
    remainingOps.push_back(op);
    originalOps.insert(op);
    return;
  });

  llvm::errs() << "filled arrays\n";

  // Clone operation by operation
  OpBuilder builder(applyOp.getBody());
  builder.setInsertionPointToStart(applyOp.getBody());

  // Keep scheduling operations
  llvm::DenseSet<Value> lifeValues;
  llvm::DenseSet<Operation *> scheduledOps;
  // Schedule all constants
  for (auto constantOp : constantOps) {
    auto clonedOp = cloneOperation(builder, constantOp.getOperation());
    updateLifeValuesAndScheduledOps(lifeValues, originalOps, scheduledOps,
                                    clonedOp);
  }
  while (!(accessOps.empty() && remainingOps.empty())) {
    // Schedule all remaining ops

    bool success = false;

    while (Operation *ready = pickReadyOp(remainingOps, lifeValues)) {
      auto clonedOp = cloneOperation(builder, ready);
      remainingOps.erase(
          std::find(remainingOps.begin(), remainingOps.end(), ready));
      updateLifeValuesAndScheduledOps(lifeValues, originalOps, scheduledOps,
                                      clonedOp);

      llvm::errs() << "clone remaining" << remainingOps.size() << "\n";
      success = true;
    }

    llvm::errs() << "#life vars " << lifeValues.size() << "\n";

    // Schedule the next access op
    if (!accessOps.empty()) {
      auto clonedOp = cloneOperation(builder, accessOps.front().getOperation());
      accessOps.erase(accessOps.begin());
      updateLifeValuesAndScheduledOps(lifeValues, originalOps, scheduledOps,
                                      clonedOp);

      llvm::errs() << "clone access" << accessOps.size() << "\n";
      success = true;
    }

    llvm::errs() << "#life vars " << lifeValues.size() << "\n";

    if (success == false) {
      applyOp.dump();
      llvm::errs() << "remaining \n";
      remainingOps.front()->dump();

      break;
    }
  }
}

} // namespace

std::unique_ptr<OpPassBase<stencil::ApplyOp>>
stencil::createStencilSchedulePass() {
  return std::make_unique<StencilSchedulePass>();
}
