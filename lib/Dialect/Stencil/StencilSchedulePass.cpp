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

// // Compute the average unroll offset
// float getAverageOffset(std::vector<Operation *> &accessOps) {
//   float result = 0.0f;
//   for (auto accessOp : accessOps) {
//     result += cast<stencil::AccessOp>(accessOp)
//                   .getOffset()[stencil::kUnrollDimension];
//   }
//   return result / (float)accessOps.size();
// }

// // Compute the life value reduction
// int64_t getLifeValueReduction(Operation *op,
//                               const std::vector<Operation *> &remainingOps,
//                               const llvm::DenseSet<Value> &lifeValues) {
//   // Check if there is a life value
//   llvm::DenseSet<Value> remainingValues;
//   for (auto remainingOp : remainingOps) {
//     if (remainingOp != op) {
//       for (auto operand : remainingOp->getOperands()) {
//         if (lifeValues.count(operand) == 1 &&
//             remainingValues.count(operand) == 0)
//           remainingValues.insert(operand);
//       }
//     }
//   }
//   return (int64_t)remainingValues.size() - (int64_t)lifeValues.size();
// }

// // Compute the distance for all operands (L2-norm)
// float getReuseDistance(OperandRange operands,
//                        std::vector<Operation *> &scheduledOps) {
//   float result = 0.0f;
//   for (auto operand : operands) {
//     // Assume constant ops have zero cost
//     if (isa_and_nonnull<ConstantOp>(operand.getDefiningOp()))
//       continue;
//     // Compute the distance for all dependencies
//     auto it = std::find(scheduledOps.begin(), scheduledOps.end(),
//                         operand.getDefiningOp());
//     if (it != scheduledOps.end()) {
//       auto distance = std::distance(it, scheduledOps.end());
//       result += (float)(distance * distance);
//     }
//   }
//   return std::sqrt(result);
// }

// // Compute operations that are ready for execution
// std::vector<Operation *>
// getReadyOps(const std::vector<Operation *> &remainingOps,
//             const llvm::DenseSet<Value> &lifeValues) {
//   std::vector<Operation *> result;
//   // Select remaining op if all parameters are life
//   for (auto remainingOp : remainingOps) {
//     if (llvm::all_of(remainingOp->getOperands(), [&](Value value) {
//           return lifeValues.count(value) != 0;
//         })) {
//       result.push_back(remainingOp);
//     }
//   }
//   return result;
// }

// // Helper method picking the next op with maximal locality
// Operation *getNextOp(const std::vector<Operation *> &remainingOps,
//                      std::vector<Operation *> &scheduledOps,
//                      const llvm::DenseSet<Value> &lifeValues) {
//   Operation *result = nullptr;
//   // Compute the candidates and pick the one that maximizes locality
//   auto candidates = getReadyOps(remainingOps, lifeValues);
//   int64_t minRed = std::numeric_limits<int64_t>::max();
//   double minDist = std::numeric_limits<float>::max();
//   for (auto candidate : candidates) {
//     int64_t candidateRed =
//         getLifeValueReduction(candidate, remainingOps, lifeValues);
//     float candidateDist =
//         getReuseDistance(candidate->getOperands(), scheduledOps);
//     if (candidateRed < minRed) {
//       minRed = candidateRed;
//       minDist = candidateDist;
//       result = candidate;
//     }
//     if (candidateRed == minRed && candidateDist < minDist) {
//       minDist = candidateDist;
//       result = candidate;
//     }
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

// Helper method to clone an operation
Operation *cloneOperation(OpBuilder &builder, Operation *op) {
  auto clonedOp = builder.clone(*op);
  for (unsigned i = 0, e = clonedOp->getNumResults(); i != e; ++i) {
    op->getResult(i).replaceAllUsesWith(clonedOp->getResult(i));
  }
  op->erase();
  return clonedOp;
}

// Class holding dependent one use operations
struct OneUseBucket {
  SmallVector<Operation *, 10> producerOps;
  SmallVector<Operation *, 10> constantOps;
  SmallVector<Operation *, 10> accessOps;
  SmallVector<Operation *, 20> arithmeticOps;
  Operation *resultOp;
  int64_t minimalOffset;
};

struct StencilSchedulePass
    : public OperationPass<StencilSchedulePass, stencil::ApplyOp> {
  void runOnOperation() override;
};

void StencilSchedulePass::runOnOperation() {
  auto applyOp = getOperation();
  // Collect all operations
  std::vector<Operation *> accessOps;
  std::vector<Operation *> constantOps;
  std::vector<Operation *> remainingOps;
  applyOp.getBody()->walk([&](Operation *op) {
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

  // Initialize the one use buckets
  std::vector<OneUseBucket> buckets;
  for (auto remainingOp : remainingOps) {
    if (!remainingOp->hasOneUse()) {
      OneUseBucket bucket = {};
      bucket.arithmeticOps = {remainingOp};
      bucket.resultOp = remainingOp;
      bucket.minimalOffset = std::numeric_limits<int64_t>::max();
      buckets.push_back(bucket);
    }
  }

  // Fill the buckets
  for (auto &bucket : buckets) {
    // Add dependencies level by level
    llvm::DenseSet<Operation *> currentLevel = {bucket.resultOp};
    llvm::DenseSet<Operation *> nextLevel; 
    SmallVector<Operation *, 10> additionalOps = {};
    do {
      // Clear collections
      additionalOps.clear();
      nextLevel.clear();
      // Compute the next level
      for (auto op : currentLevel) {
        for (auto operand : op->getOperands()) {
          if (operand.getDefiningOp())
            nextLevel.insert(operand.getDefiningOp());
        }
      }
      // Store the current level
      for (auto op : nextLevel) {
        if (isa<ConstantOp>(op)) {
          bucket.constantOps.push_back(op);
          continue;
        }
        if (isa<stencil::AccessOp>(op)) {
          bucket.accessOps.push_back(op);
          continue;
        }
        if (!op->hasOneUse()) {
          bucket.producerOps.push_back(op);
          continue;
        }
        additionalOps.push_back(op);
      }
      // Add the arithmetic ops
      bucket.arithmeticOps.insert(bucket.arithmeticOps.begin(),
                                  additionalOps.begin(), additionalOps.end());
      // Update the current level
      currentLevel.clear();
      currentLevel.insert(additionalOps.begin(), additionalOps.end());
    } while (!additionalOps.empty());
    // Compute the minimal offset
    for (auto accessOp : bucket.accessOps) {
      bucket.minimalOffset = std::min(
          bucket.minimalOffset, cast<stencil::AccessOp>(accessOp)
                                    .getOffset()[stencil::kUnrollDimension]);
    }

    // llvm::errs() << "- bucket\n";
    // llvm::errs() << "  -> accessOps" << bucket.accessOps.size() << "\n";
    // llvm::errs() << "  -> constantOps" << bucket.constantOps.size() << "\n";
    // llvm::errs() << "  -> arithmeticOps" << bucket.arithmeticOps.size() << "\n";
    // llvm::errs() << "  -> producerOps" << bucket.producerOps.size() << "\n";
  }

  // Clone operation by operation
  OpBuilder builder(applyOp.getBody());

  // Keep scheduling operations
  //llvm::DenseSet<Value> lifeValues;
  llvm::DenseSet<Operation*> scheduledOps;
  //std::map<size_t, unsigned> lifeFrequencies;
  // Schedule all constants
  for (auto constantOp : constantOps) {
    auto clonedOp = cloneOperation(builder, constantOp);
    scheduledOps.insert(constantOp);

    //updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
    //lifeFrequencies[lifeValues.size()]++;
  }

  // Schedule all buckets
  for (auto &bucket : buckets) {
    // Schedule the access ops
    for (auto accessOp : bucket.accessOps) {
      //llvm::errs() << "accessOp " << accessOp << "\n";
      if (scheduledOps.count(accessOp) == 0) {
        auto clonedOp = cloneOperation(builder, accessOp);
        //llvm::errs() << "clonedOp " << clonedOp << "\n";
        scheduledOps.insert(accessOp);
        // updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
        // lifeFrequencies[lifeValues.size()]++;
        //llvm::errs() << "cloning\n";
      }
    }

    // ASSERT all poducers
    // (alread scheduled...)
    // for(auto producerOp : bucket.producerOps) {
    //   assert(scheduledOps.count(producerOp) != 0 && "expect producerOp to be scheduled");
    // }
    // for(auto accessOp : bucket.accessOps) {
    //   llvm::errs() << "asserting accessOp " << accessOp << "\n";
    //   assert(scheduledOps.count(accessOp) != 0 && "expect accessOp to be scheduled");
    // }

    // Schedule the access ops
    for (auto arithmeticOp : bucket.arithmeticOps) {
      auto clonedOp = cloneOperation(builder, arithmeticOp);
      scheduledOps.insert(arithmeticOp);

      // updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
      // lifeFrequencies[lifeValues.size()]++;

      //llvm::outs() << "// life values " << lifeValues.size() << "\n";
    }
  }

  //TODO order the stuff properly

  //applyOp.dump();

  // #ifdef NO_SCHEDULER
  // std::vector<Operation *> remainingOps;
  // applyOp.getBody()->walk([&](Operation *op) {
  //   // Store all remaining ops
  //   remainingOps.push_back(op);
  // });

  // // Clone operation by operation
  // OpBuilder builder(applyOp.getBody());

  // llvm::DenseSet<Value> lifeValues;
  // std::vector<Operation *> scheduledOps;
  // std::map<size_t, unsigned> lifeFrequencies;

  // for (auto constantOp : remainingOps) {
  //   // auto clonedOp = cloneOperation(builder, constantOp);
  //   // updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
  //   // lifeFrequencies[lifeValues.size()]++;

  //   //constantOp->dump();
  //   // detect one use groups
  //   if(!isa<ConstantOp>(constantOp) ) {
  //     llvm::outs() << "life values " << lifeValues.size() << " has one " <<
  //     constantOp->hasOneUse() << "\n";

  //     if(!constantOp->hasOneUse())
  //       constantOp->dump();
  //   }

  // }
  // #endif

  // // Print the top5 life value frequencies
  // llvm::outs() << "// Life Value Frequencies\n"
  //              << "// ======================\n";
  // for (auto it = lifeFrequencies.begin(); it != lifeFrequencies.end(); ++it)
  //   llvm::outs() << "// - " << it->first << "(" << it->second << ")\n";
}

} // namespace

std::unique_ptr<OpPassBase<stencil::ApplyOp>>
stencil::createStencilSchedulePass() {
  return std::make_unique<StencilSchedulePass>();
}

void stencil::createStencilSchedulePipeline(OpPassManager &pm) {
  auto &funcPm = pm.nest<FuncOp>();
  // funcPm.addPass(createStencilPreShufflePass());
  // funcPm.addPass(createStencilShufflePass());
  // funcPm.addPass(createStencilPostShufflePass());
  funcPm.addPass(createStencilSchedulePass());
}

void stencil::createStencilScheduleOnlyPipeline(OpPassManager &pm) {
  auto &funcPm = pm.nest<FuncOp>();
  // funcPm.addPass(createStencilShufflePass());
  funcPm.addPass(createStencilSchedulePass());
}

// // Walk all operations and store constants and accesses separately
// std::vector<Operation *> accessOps;
// std::vector<Operation *> constantOps;
// std::vector<Operation *> remainingOps;
// applyOp.getBody()->walk(
//     [&accessOps, &constantOps, &remainingOps](Operation *op) {
//       // Store all access ops
//       if (isa<stencil::AccessOp>(op)) {
//         accessOps.push_back(op);
//         return;
//       }
//       // Store all constant ops
//       if (isa<ConstantOp>(op)) {
//         constantOps.push_back(op);
//         return;
//       }
//       // Store all remaining ops
//       remainingOps.push_back(op);
//     });

// // Clone operation by operation
// OpBuilder builder(applyOp.getBody());

// // Keep scheduling operations
// llvm::DenseSet<Value> lifeValues;
// std::vector<Operation *> scheduledOps;
// std::map<size_t, unsigned> lifeFrequencies;
// // Schedule all constants
// for (auto constantOp : constantOps) {
//   auto clonedOp = cloneOperation(builder, constantOp);
//   updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
//   lifeFrequencies[lifeValues.size()]++;
// }
// // Schedule accesses and arithmetic Ops
// while (!(accessOps.empty() && remainingOps.empty())) {
//   // Schedule the next operation with maximal locality
//   // (If multiple operations have the same )
//   while (Operation *next =
//              getNextOp(remainingOps, scheduledOps, lifeValues)) {
//     auto clonedOp = cloneOperation(builder, next);
//     updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
//     remainingOps.erase(llvm::find(remainingOps, next));
//     lifeFrequencies[lifeValues.size()]++;
//     llvm::outs() << "life values " << lifeValues.size() << " is div " <<
//     isa<DivFOp>(next) << "\n";
//   }

//   // Compute accessOp / arithmeticOp pairs
//   typedef std::vector<Operation *> OpList;
//   typedef std::tuple<OpList, int64_t, float, float> Candidate;
//   std::vector<Candidate> candidateOps;
//   for (auto remainingOp : remainingOps) {
//     // Test scheduling the access op
//     OpList scheduled = scheduledOps;
//     llvm::DenseSet<Value> life = lifeValues;
//     // Compute the dependencies needed to schedule the op
//     OpList dependencies;
//     for (auto operand : remainingOp->getOperands()) {
//       if (operand.getDefiningOp() &&
//           !llvm::is_contained(scheduledOps, operand.getDefiningOp())) {
//         dependencies.push_back(operand.getDefiningOp());
//       }
//     }
//     // Verify all dependencies are access ops
//     if (llvm::any_of(dependencies, [&](Operation *op) {
//           return !llvm::is_contained(accessOps, op);
//         }))
//       continue;
//     std::sort(dependencies.begin(), dependencies.end(),
//               [](Operation *op1, Operation *op2) {
//                 return cast<stencil::AccessOp>(op1)
//                            .getOffset()[stencil::kUnrollDimension] <
//                        cast<stencil::AccessOp>(op2)
//                            .getOffset()[stencil::kUnrollDimension];
//               });
//     // Test schedule the access dependencies
//     for (auto dependency : dependencies) {
//       scheduled.push_back(dependency);
//       life.insert(dependency->getResult(0));
//     }
//     // Get the next operation
//     Operation *next = getNextOp(remainingOps, scheduled, life);
//     if (next) {
//       int64_t lifeValueReduction =
//           getLifeValueReduction(next, remainingOps, lifeValues);
//       float unrollOffset = getAverageOffset(dependencies);
//       float reuseDistance = getReuseDistance(next->getOperands(), scheduled);
//       candidateOps.push_back(std::make_tuple(dependencies,
//       lifeValueReduction,
//                                              reuseDistance, unrollOffset));
//     }
//   }

//   // TODO consider reduction of life values!!!

//   // Sort the access op candidates
//   // pick the one with minimal unroll index
//   std::sort(candidateOps.begin(), candidateOps.end(),
//             [](Candidate x, Candidate y) {
//               return std::get<3>(x) < std::get<3>(y);
//               //  ||
//               //        (std::get<1>(x) == std::get<1>(y) &&
//               //         std::get<3>(x) < std::get<3>(y));
//               //       //    ||
//                     //  (std::get<1>(x) == std::get<1>(y) &&
//                     //   std::get<2>(x) == std::get<2>(y) &&
//                     //   std::get<3>(x) < std::get<3>(y));
//             });

//   // print the candidates
//   // llvm::outs() << "candidates : \n";
//   // for (auto candidateOp : candidateOps)
//   //   llvm::outs() << " - cand " << std::get<1>(candidateOp) << " - "
//   //                << std::get<2>(candidateOp) << " - "
//   //                << std::get<3>(candidateOp) << "\n";

//   if (!candidateOps.empty()) {
//     for (auto candidateOp : std::get<0>(candidateOps.front())) {
//       auto clonedOp = cloneOperation(builder, candidateOp);
//       updateLifeValuesAndScheduledOps(lifeValues, scheduledOps, clonedOp);
//       accessOps.erase(llvm::find(accessOps, candidateOp));
//       lifeFrequencies[lifeValues.size()]++;
//     }
//   }
// }