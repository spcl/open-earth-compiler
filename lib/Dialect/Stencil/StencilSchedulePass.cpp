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
#include <algorithm>
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>
#include <limits>
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

// Compute the
float getAverageOffset(std::vector<Operation *> &accessOps) {
  float result = 0.0f;
  for (auto accessOp : accessOps) {
    result += cast<stencil::AccessOp>(accessOp)
                  .getOffset()[stencil::kUnrollDimension];
  }
  return result / (float)accessOps.size();
}

// Compute the distance for all operands (L2-norm)
float getReuseDistance(OperandRange operands,
                       std::vector<Operation *> &scheduledOps) {
  float result = 0.0f;
  for (auto operand : operands) {
    // Assume constant ops have zero cost
    if (isa_and_nonnull<ConstantOp>(operand.getDefiningOp()))
      continue;
    // Compute the distance for all dependencies
    auto it = std::find(scheduledOps.begin(), scheduledOps.end(),
                        operand.getDefiningOp());
    if (it != scheduledOps.end()) {
      auto distance = std::distance(it, scheduledOps.end());
      result += (float)(distance * distance);
    }
  }
  return std::sqrt(result);
}

// Compute operations that are ready for execution
std::vector<Operation *>
getReadyOps(const std::vector<Operation *> &remainingOps,
            const llvm::DenseSet<Value> &lifeValues) {
  std::vector<Operation *> result;
  // Select remaining op if all parameters are life
  for (auto remainingOp : remainingOps) {
    if (llvm::all_of(remainingOp->getOperands(), [&](Value value) {
          return lifeValues.count(value) != 0;
        })) {
      result.push_back(remainingOp);
    }
  }
  return result;
}

// Helper method picking the next op with maximal locality
Operation *getOpWithMaxLoc(const std::vector<Operation *> &remainingOps,
                           std::vector<Operation *> &scheduledOps,
                           const llvm::DenseSet<Value> &lifeValues) {
  Operation *result = nullptr;
  // Compute the candidates and pick the one that maximizes locality
  auto candidates = getReadyOps(remainingOps, lifeValues);
  double minDist = std::numeric_limits<float>::max();
  for (auto candidate : candidates) {
    float candidateDist =
        getReuseDistance(candidate->getOperands(), scheduledOps);
    if (candidateDist < minDist) {
      minDist = candidateDist;
      result = candidate;
    }
  }
  return result;
}

// // Helper method picking the next operation to schedule
// Operation *getNextReadyOp(const std::vector<Operation *> &remainingOps,
//                           const llvm::DenseSet<Value> &lifeValues) {
//   // Select remaining op if all parameters are life
//   for (auto remainingOp : remainingOps) {
//     if (llvm::all_of(remainingOp->getOperands(), [&](Value value) {
//           return lifeValues.count(value) != 0;
//         })) {
//       return remainingOp;
//     }
//   }
//   return nullptr;
// }

// // Helper method to count the number of ready ops
// unsigned countReadyOps(const std::vector<Operation *> &remainingOps,
//                        const llvm::DenseSet<Value> &lifeValues) {
//   unsigned result = 0;
//   // Select remaining op if all parameters are life
//   for (auto remainingOp : remainingOps) {
//     if (llvm::all_of(remainingOp->getOperands(),
//                      [&](Value value) { return lifeValues.count(value) != 0;
//                      }))
//       result++;
//   }
//   return result;
// }

void updateLifeValuesAndScheduledOps(llvm::DenseSet<Value> &lifeValues,
                                     std::vector<Operation *> &scheduledOps,
                                     Operation *clonedOp) {
  // Add all results to life values
  for (auto result : clonedOp->getResults()) {
    lifeValues.insert(result);
  }
  scheduledOps.push_back(clonedOp);

  // Update life values
  for (auto value : clonedOp->getOperands()) {
    if (llvm::all_of(value.getUsers(), [&](Operation *user) {
          return llvm::is_contained(scheduledOps, user);
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
  std::vector<Operation *> scheduledOps;
  std::map<size_t, unsigned> lifeFrequencies;
  // Schedule all constants
  for (auto constantOp : constantOps) {
    auto clonedOp = cloneOperation(builder, constantOp);
    updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
    lifeFrequencies[lifeValues.size()]++;
  }
  // Schedule accesses and arithmetic Ops
  while (!(accessOps.empty() && remainingOps.empty())) {
    // Schedule the next operation with maximal locality
    // (If multiple operations have the same )
    while (Operation *next =
               getOpWithMaxLoc(remainingOps, scheduledOps, lifeValues)) {
      auto clonedOp = cloneOperation(builder, next);
      updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
      remainingOps.erase(llvm::find(remainingOps, next));
      lifeFrequencies[lifeValues.size()]++;
    }

    // Schedule all arithmetic Ops that are ready for execution
    // while (Operation *next = getNextReadyOp(remainingOps, lifeValues)) {
    //   auto clonedOp = cloneOperation(builder, next);
    //   updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
    //   remainingOps.erase(llvm::find(remainingOps, next));
    //   lifeFrequencies[lifeValues.size()]++;
    // }

    // Compute accessOp / arithmeticOp pairs
    typedef std::tuple<std::vector<Operation *>, float, float> Candidate;
    std::vector<Candidate> candidateOps;
    for (auto remainingOp : remainingOps) {
      // Test scheduling the access op
      std::vector<Operation *> scheduled = scheduledOps;
      llvm::DenseSet<Value> life = lifeValues;
      // Compute the dependencies needed to schedule the op
      std::vector<Operation *> dependencies;
      for (auto operand : remainingOp->getOperands()) {
        if (operand.getDefiningOp() &&
            !llvm::is_contained(scheduledOps, operand.getDefiningOp())) {
          dependencies.push_back(operand.getDefiningOp());
        }
      }
      // Verify all dependencies are access ops
      if (llvm::any_of(dependencies, [&](Operation *op) {
            return !llvm::is_contained(accessOps, op);
          }))
        continue;
      std::sort(dependencies.begin(), dependencies.end(), [](Operation* op1, Operation* op2) {
        return cast<stencil::AccessOp>(op1).getOffset()[stencil::kUnrollDimension] <
              cast<stencil::AccessOp>(op2).getOffset()[stencil::kUnrollDimension];
      });
      // Test schedule the access dependencies
      for (auto dependency : dependencies) {
        scheduled.push_back(dependency);
        life.insert(dependency->getResult(0));
      }
      // Get the next operation
      Operation *next = getOpWithMaxLoc(remainingOps, scheduled, life);
      if (next) {
        float reuseDistance = getReuseDistance(next->getOperands(), scheduled);
        float unrollOffset = getAverageOffset(dependencies);
        candidateOps.push_back(
            std::make_tuple(dependencies, unrollOffset, reuseDistance));
      }
    }

    // TODO consider reduction of life values!!!
    
    // Sort the access op candidates
    // pick the one with minimal unroll index
    std::sort(candidateOps.begin(), candidateOps.end(),
              [](Candidate x, Candidate y) {
                return std::get<1>(x) < std::get<1>(y) ||
                       (std::get<1>(x) == std::get<1>(y) &&
                        std::get<2>(x) < std::get<2>(y));
              });

    // // print the candidates
    // llvm::outs() << "candidates : \n";
    // for (auto candidateOp : candidateOps)
    //   llvm::outs() << " - op loc "
    //                << std::get<1>(candidateOp) << " off "
    //                << std::get<2>(candidateOp) << "\n";

    if (!candidateOps.empty()) {
      for(auto candidateOp : std::get<0>(candidateOps.front())) {
        auto clonedOp = cloneOperation(builder, candidateOp);
        updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
        accessOps.erase(llvm::find(accessOps, candidateOp));
        lifeFrequencies[lifeValues.size()]++;
      }
    }

    // // Compute list of operands needed to get ops ready
    // llvm::DenseSet<Value> candidateValues;
    // llvm::DenseSet<Operation *> predecessorOps;
    // for (auto remainingOp : remainingOps) {
    //   if (llvm::none_of(remainingOp->getOperands(), [&](Value value) {
    //         return predecessorOps.count(value.getDefiningOp()) != 0;
    //       })) {
    //     for (auto operand : remainingOp->getOperands()) {
    //       candidateValues.insert(operand);
    //     }
    //   }
    //   // Keep track of earlier operations
    //   predecessorOps.insert(remainingOp);
    // }

    // // Compute list of operands needed to get ops ready
    // llvm::DenseSet<Value> candidateValues;
    // llvm::DenseSet<Operation *> predecessorOps;
    // for (auto remainingOp : remainingOps) {
    //   if (llvm::none_of(remainingOp->getOperands(), [&](Value value) {
    //         return predecessorOps.count(value.getDefiningOp()) != 0;
    //       })) {
    //     for (auto operand : remainingOp->getOperands()) {
    //       candidateValues.insert(operand);
    //     }
    //   }
    //   // Keep track of earlier operations
    //   predecessorOps.insert(remainingOp);
    // }

    // Consider locality of the remaining op
    // Consider unroll coordinate
    // (if similar locality choose min coordinate)

    // Consider Locality
    // TODO sort by locality & then j-index

    // Local  ops

    // std::vector<stencil::AccessOp> candidates;
    // for (auto it = accessOps.begin(); it != accessOps.end(); ++it) {
    //   if (candidateValues.count((*it)->getResult(0)) != 0) {
    //     candidates.push_back(cast<stencil::AccessOp>(*it));
    //   }

    //   // if(llvm::is_contained(remainingOps.front()->getOperands(),
    //   // (*it)->getResult(0))) {
    //   //   auto clonedOp = cloneOperation(builder, *it);
    //   //   updateLifeValuesAndScheduledOps(lifeValues, scheduledOps,
    //   clonedOp);
    //   //   accessOps.erase(it);
    //   //   lifeFrequencies[lifeValues.size()]++;
    //   //   break;
    //   // }
    // }
    // // pick the one with minimal unroll index
    // std::sort(candidates.begin(), candidates.end(),
    //           [](stencil::AccessOp x, stencil::AccessOp y) {
    //             return x.getOffset()[stencil::kUnrollDimension] <
    //                    y.getOffset()[stencil::kUnrollDimension];
    //           });

    // schedule local computation
    // TODO
    // (if there is nothing go choose the next bulk of work purely by j index)

    // for (auto candidate : candidates)
    //   llvm::outs() << candidate.getOffset()[stencil::kUnrollDimension] << ",
    //   ";
    // llvm::outs() << "\n";

    // if (!candidates.empty()) {
    //   auto clonedOp =
    //       cloneOperation(builder, candidates.front().getOperation());
    //   updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
    //   accessOps.erase(llvm::find(accessOps,
    //   candidates.front().getOperation()));
    //   lifeFrequencies[lifeValues.size()]++;
    // }
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
  funcPm.addPass(createStencilPostShufflePass());
  funcPm.addPass(createStencilSchedulePass());
}

void stencil::createStencilScheduleOnlyPipeline(OpPassManager &pm) {
  auto &funcPm = pm.nest<FuncOp>();
  // funcPm.addPass(createStencilShufflePass());
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
//     //       llvm::outs() << "    -> " << bucket.first[0] << "," <<
//     bucket.first[1] << "," << bucket.first[2] << "\n";
//     //       llvm::outs() << "    -> bucket size " << bucket.second.size() <<
//     "\n";
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