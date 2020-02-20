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

// Helper method to count the number of ready ops
unsigned countReadyOps(const std::vector<Operation *> &remainingOps,
                       const llvm::DenseSet<Value> &lifeValues) {
  unsigned result = 0;
  // Select remaining op if all parameters are life
  for (auto remainingOp : remainingOps) {
    if (llvm::all_of(remainingOp->getOperands(),
                     [&](Value value) { return lifeValues.count(value) != 0; }))
      result++;
  }
  return result;
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
  std::vector<Operation *> remainingOps;
  applyOp.getBody()->walk(
      [&accessOps, &constantOps, &remainingOps](Operation *op) {
        // Store all access ops
        if (isa<stencil::AccessOp>(op)) {
          accessOps.push_back(op);
          return;
        }
        // Store all constant ops
        if (isa<ConstantOp>(op)) {
          constantOps.push_back(op);
          return;
        }
        // Store all remaining ops
        remainingOps.push_back(op);
      });

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
  while (!(accessOps.empty() && remainingOps.empty())) {
    // Schedule all arithmetic Ops that are ready for execution
    while (Operation *next = getNextReadyOp(remainingOps, lifeValues)) {
      auto clonedOp = cloneOperation(builder, next);
      updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
      remainingOps.erase(llvm::find(remainingOps, next));
      lifeFrequencies[lifeValues.size()]++;
    }

    // Schedule the access ops needed to release the next arithmetic op
    for(auto it = accessOps.begin(); it != accessOps.end(); ++it) {
      if(llvm::is_contained(remainingOps.front()->getOperands(), (*it)->getResult(0))) {
        auto clonedOp = cloneOperation(builder, *it);
        updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
        accessOps.erase(it);
        lifeFrequencies[lifeValues.size()]++;
        break;
      }
    }
  }

  // Print the top5 life value frequencies
  llvm::outs() << "// Life Value Frequencies\n"
               << "// ======================\n";
  for (auto it = lifeFrequencies.begin(); it != lifeFrequencies.end(); ++it)
    llvm::outs() << "// - " << it->first << "(" << it->second << ")\n";
}

} // namespace

std::unique_ptr<OpPassBase<stencil::ApplyOp>>
stencil::createStencilSchedulePass() {
  return std::make_unique<StencilSchedulePass>();
}

void stencil::createStencilSchedulePipeline(OpPassManager &pm) {
  auto &funcPm = pm.nest<FuncOp>();
  // funcPm.addPass(createStencilPreShufflePass());
  funcPm.addPass(createStencilShufflePass());
  funcPm.addPass(createStencilSchedulePass());
  funcPm.addPass(createStencilPostShufflePass());
}

void stencil::createStencilScheduleOnlyPipeline(OpPassManager &pm) {
  auto &funcPm = pm.nest<FuncOp>();
  funcPm.addPass(createStencilShufflePass());
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


// OLD SCHEDULING
// void StencilSchedulePass::runOnOperation() {
//   auto applyOp = getOperation();
//   // Walk all operations and store constants and accesses separately
//   std::vector<Operation *> constantOps;
//   std::vector<Operation *> remainingOps;
//   // Store the access ops of every field in buckets with the same j-Offset
//   typedef SmallVector<int64_t, 3> Offset;
//   typedef std::vector<Operation *> OpList;
//   typedef std::pair<Offset, OpList> Bucket;
//   llvm::DenseMap<Value, std::map<int64_t, std::vector<Bucket>>> accessOps;
//   applyOp.getBody()->walk(
//       [&accessOps, &constantOps, &remainingOps](Operation *op) {
//         // Store all access ops
//         if (auto accessOp = dyn_cast<stencil::AccessOp>(op)) {
//           //auto idx = accessOp.getOffset()[stencil::kUnrollDimension];
//           auto idx = 0;
//           auto &buckets = accessOps[accessOp.getOperand()][idx];
//           // Search the bucket for entry with same index
//           auto it = std::find_if(buckets.begin(), buckets.end(),
//                                  [&accessOp](auto bucket) {
//                                    Offset offset = accessOp.getOffset();
//                                    offset[stencil::kVectorDimension] =
//                                        bucket.first[stencil::kVectorDimension];
//                                    return offset == bucket.first;
//                                  });
//           if (it != buckets.end()) {
//             it->second.push_back(op);
//           } else {
//             buckets.push_back(
//                 std::make_pair<Offset, OpList>(accessOp.getOffset(), {op}));
//           }
//           return;
//         }
//         // Store all constant ops
//         if (isa<ConstantOp>(op)) {
//           constantOps.push_back(op);
//           return;
//         }
//         // Store all remaining ops
//         remainingOps.push_back(op);
//       });

//   // Clone operation by operation
//   OpBuilder builder(applyOp.getBody());

//   // Keep scheduling operations
//   llvm::DenseSet<Value> lifeValues;
//   llvm::DenseSet<Operation *> scheduledOps;
//   std::map<size_t, unsigned> lifeFrequencies;
//   // Schedule all constants
//   for (auto constantOp : constantOps) {
//     auto clonedOp = cloneOperation(builder, constantOp);
//     updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
//     lifeFrequencies[lifeValues.size()]++;
//   }
//   // Schedule accesses and arithmetic Ops
//   while (!(accessOps.empty() && remainingOps.empty())) {
//     // Schedule all arithmetic Ops that are ready for execution
//     while (Operation *next = getNextReadyOp(remainingOps, lifeValues)) {
//       auto clonedOp = cloneOperation(builder, next);
//       updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
//       remainingOps.erase(llvm::find(remainingOps, next));
//       lifeFrequencies[lifeValues.size()]++;
//     }

//     // Schedule the access op bucket that enables most arithmetic ops
//     unsigned maxReadyOpCount = 0;
//     Value maxValue(nullptr);
//     size_t maxIdx;
//     for (auto value : accessOps) {
//       // Compute the number of ready ops for all buckets with minimal index
//       auto &buckets = value.getSecond().begin()->second;
//       for (auto it = buckets.begin(); it != buckets.end(); ++it) {
//         // Compute the next life values
//         llvm::DenseSet<Value> nextLifeValues = lifeValues;
//         for (auto op : it->second) {
//           nextLifeValues.insert(op->getResult(0));
//         }
//         unsigned readyOpCount = countReadyOps(remainingOps, nextLifeValues);
//         if (readyOpCount >= maxReadyOpCount) {
//           maxReadyOpCount = readyOpCount;
//           maxValue = value.getFirst();
//           maxIdx = std::distance(buckets.begin(), it);
//         }
//       }
//     }

//     // Schedule the next access operations
//     if (maxValue) {
//       // Clone all ops of the bucket
//       auto &buckets = accessOps[maxValue].begin()->second;
//       for(auto op : buckets[maxIdx].second) {
//         auto clonedOp = cloneOperation(builder, op);
//         updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);        
//       }
//       // Erase the bucket from the bucket data structure
//       buckets.erase(buckets.begin() + maxIdx);
//       if(buckets.empty()) {
//         accessOps[maxValue].erase(accessOps[maxValue].begin()->first);
//         if(accessOps[maxValue].empty()) {
//           accessOps.erase(maxValue);
//         }
//       }
//       lifeFrequencies[lifeValues.size()]++;
//     }

//     // // Print the access ops
//     // llvm::outs() << "-> accessOps\n";
//     // for(auto value : accessOps) {
//     //   llvm::outs() << "  -> " << value.getFirst() << "\n";
//     //   for(auto index : value.getSecond()) {
//     //     llvm::outs() << "    -> " << index.first << "\n";
//     //     for(auto bucket : index.second) {
//     //       llvm::outs() << "    -> " << bucket.first[0] << "," << bucket.first[1] << "," << bucket.first[2] << "\n";
//     //       llvm::outs() << "    -> bucket size " << bucket.second.size() << "\n";
//     //       for(auto op : bucket.second)
//     //         llvm::outs() << "    -> " << op << "\n";
//     //     }
//     //   }
//     // }

//   }

//   // Print the top5 life value frequencies
//   llvm::outs() << "// Life Value Frequencies\n"
//                << "// ======================\n";
//   for (auto it = lifeFrequencies.begin(); it != lifeFrequencies.end(); ++it)
//     llvm::outs() << "// - " << it->first << "(" << it->second << ")\n";
// }