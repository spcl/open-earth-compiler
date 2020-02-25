#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTraits.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
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
#include "llvm/ADT/DenseMap.h"
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

bool compareOperations(Operation *op1, Operation *op2) {
  // Sort by view first
  auto view1 = cast<stencil::AccessOp>(op1).getOperand().getAsOpaquePointer();
  auto view2 = cast<stencil::AccessOp>(op2).getOperand().getAsOpaquePointer();
  if (view1 == view2) {
    // Sort by offset otherwise
    SmallVector<int64_t, 3> offset1 = cast<stencil::AccessOp>(op1).getOffset();
    SmallVector<int64_t, 3> offset2 = cast<stencil::AccessOp>(op2).getOffset();
    return (offset1[stencil::kUnrollDimension] <
            offset2[stencil::kUnrollDimension]) ||
           (offset1[stencil::kUnrollDimension] ==
                offset2[stencil::kUnrollDimension] &&
            offset1[stencil::kKDimension] < offset2[stencil::kKDimension]) ||
           (offset1[stencil::kUnrollDimension] ==
                offset2[stencil::kUnrollDimension] &&
            offset1[stencil::kKDimension] == offset2[stencil::kKDimension] &&
            offset1[stencil::kVectorDimension] <
                offset2[stencil::kVectorDimension]);
  }
  return view1 < view2;
};

// Helper that follows all dependencies down to the access offsets
void collectAccessPattern(Operation *op,
                          SmallVector<Operation *, 20> &accessOps) {
  if (auto accessOp = dyn_cast_or_null<stencil::AccessOp>(op)) {
    accessOps.push_back(op);
  } else {
    for (auto operand : op->getOperands()) {
      collectAccessPattern(operand.getDefiningOp(), accessOps);
    }
  }
}

bool arePatternEqual(SmallVector<Operation *, 20> &pattern1,
                     SmallVector<Operation *, 20> &pattern2) {
  // Check they have the same size
  if (pattern1.size() != pattern2.size())
    return false;
  // Check they access the same field
  if (!llvm::all_of(
          llvm::zip(pattern1, pattern2),
          [](std::tuple<Operation *, Operation *> x) {
            return std::get<0>(x)->getOperand(0).getAsOpaquePointer() ==
                   std::get<1>(x)->getOperand(0).getAsOpaquePointer();
          }))
    return false;
  // Compute the access offsets
  if (pattern1.empty())
    return false;
  // Compute the shift for the first access
  auto computeShift = [](Operation *op1, Operation *op2) {
    SmallVector<int64_t, 3> result;
    auto offset1 = cast<stencil::AccessOp>(op1).getOffset();
    auto offset2 = cast<stencil::AccessOp>(op2).getOffset();
    llvm::transform(llvm::zip(offset1, offset2), result.begin(),
                    [](std::tuple<int64_t, int64_t> x) {
                      return std::get<1>(x) - std::get<0>(x);
                    });
    return result;
  };
  SmallVector<int64_t, 3> shift = computeShift(pattern1[0], pattern2[0]);
  return llvm::all_of(llvm::zip(pattern1, pattern2),
                      [&](std::tuple<Operation *, Operation *> x) {
                        return shift ==
                               computeShift(std::get<0>(x), std::get<1>(x));
                      });
};

// Class holding dependent one use operations
struct Bucket {
  SmallVector<Operation *, 10> producerOps;
  SmallVector<Operation *, 10> constantOps;
  SmallVector<Operation *, 10> accessOps;
  SmallVector<Operation *, 20> internalOps;
  Operation *rootOp;
  int64_t minOffset;
};

// Helper method generating a bucket given the root operation
Bucket createBucket(Operation *op) {
  Bucket bucket = {};
  bucket.internalOps = {op};
  bucket.rootOp = op;
  bucket.minOffset = std::numeric_limits<int64_t>::max();
  return bucket;
}

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
  assert(dyn_cast<stencil::ReturnOp>(remainingOps.back()) &&
         "expected stencil return op");
  std::vector<Bucket> buckets = {createBucket(remainingOps.back())};

  // Fill the buckets
  for (unsigned i = 0; i < buckets.size(); ++i) {
    // Add dependencies level by level
    SmallVector<Operation *, 10> currentLevel = {buckets[i].rootOp};
    SmallVector<Operation *, 10> internalOps = {};
    do {
      // Clear collections
      internalOps.clear();
      llvm::DenseSet<Operation *> nextLevel;
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
          buckets[i].constantOps.push_back(op);
          continue;
        }
        if (isa<stencil::AccessOp>(op)) {
          buckets[i].accessOps.push_back(op);
          continue;
        }
        if (!op->hasOneUse()) {
          buckets[i].producerOps.push_back(op);
          if (llvm::none_of(buckets, [&op](Bucket &bucket) {
                return bucket.rootOp == op;
              })) {
            buckets.push_back(createBucket(op));
          }
          continue;
        }
        internalOps.push_back(op);
      }

      // Check if there are stencils in the next level
      llvm::DenseMap<Operation *, SmallVector<Operation *, 20>> accessPatterns;
      // Compute all access patterns
      for (auto internalOp : internalOps) {
        SmallVector<Operation *, 20> accessOps;
        collectAccessPattern(internalOp, accessOps);
        std::sort(accessOps.begin(), accessOps.end());
        auto it = std::unique(accessOps.begin(), accessOps.end());
        accessOps.erase(it, accessOps.end());
        std::sort(accessOps.begin(), accessOps.end(), compareOperations);
        accessPatterns[internalOp] = accessOps;
      }
      // Search for equal patterns and make them producer ops
      SmallVector<Operation *, 10> equalOps;
      do {
        equalOps.clear();
        for (int64_t i1 = 0; i1 < (int64_t)internalOps.size() - 1; ++i1) {
          equalOps.clear();
          for (int64_t i2 = i1 + 1; i2 < (int64_t)internalOps.size(); ++i2) {
            if (arePatternEqual(accessPatterns[internalOps[i1]],
                                accessPatterns[internalOps[i2]]))
              equalOps.push_back(internalOps[i2]);
          }
          // Update the additional ops
          if (!equalOps.empty()) {
            equalOps.push_back(internalOps[i1]);
            for (auto equalOp : equalOps) {
              buckets[i].producerOps.push_back(equalOp);
              if (llvm::none_of(buckets, [&equalOp](Bucket &bucket) {
                    return bucket.rootOp == equalOp;
                  })) {
                buckets.push_back(createBucket(equalOp));
              }
              auto it = llvm::find(internalOps, equalOp);
              internalOps.erase(it);
            }
            break;
          }
        }
      } while (!equalOps.empty());

      // Add the arithmetic ops
      buckets[i].internalOps.insert(buckets[i].internalOps.begin(),
                                    internalOps.begin(), internalOps.end());
      // Update the current level
      currentLevel = internalOps;
    } while (!internalOps.empty());

    // Compute the minimal offset
    for (auto accessOp : buckets[i].accessOps) {
      buckets[i].minOffset = std::min(
          buckets[i].minOffset, cast<stencil::AccessOp>(accessOp)
                                    .getOffset()[stencil::kUnrollDimension]);
    }
    // Sort the access ops
    std::sort(buckets[i].accessOps.begin(), buckets[i].accessOps.end(),
              compareOperations);

    // llvm::errs() << "- bucket\n";
    // llvm::errs() << "  -> accessOps" << buckets[i].accessOps.size() << "\n";
    // llvm::errs() << "  -> constantOps" << buckets[i].constantOps.size() <<
    // "\n"; llvm::errs() << "  -> internalOps" << buckets[i].internalOps.size()
    // << "\n"; llvm::errs() << "  -> producerOps" <<
    // buckets[i].producerOps.size() << "\n";
  }

  // Clone operation by operation
  OpBuilder builder(applyOp.getBody());

  // Keep scheduling operations
  llvm::DenseSet<Operation *> scheduledOps;
  // Schedule all constants
  for (auto constantOp : constantOps) {
    auto clonedOp = cloneOperation(builder, constantOp);
    scheduledOps.insert(constantOp);
  }

  // Schedule all buckets by offset
  // while (!buckets.empty()) {
  //   // Find the bucket with the minimal offset
  //   int64_t minOffset = std::numeric_limits<int64_t>::max();
  //   std::vector<Bucket>::iterator minIt;
  //   for (auto it = buckets.begin(); it != buckets.end(); ++it) {
  //     if (it->minOffset <= minOffset &&
  //         llvm::all_of(it->producerOps, [&](Operation *op) {
  //           return scheduledOps.count(op) != 0;
  //         })) {
  //       minOffset = it->minOffset;
  //       minIt = it;
  //     }
  //   }

  //   // Schedule the actual bucket
  //   for (auto accessOp : minIt->accessOps) {
  //     if (scheduledOps.count(accessOp) == 0) {
  //       auto clonedOp = cloneOperation(builder, accessOp);
  //       scheduledOps.insert(accessOp);
  //     }
  //   }
  //   for (auto internalOp : minIt->internalOps) {
  //     auto clonedOp = cloneOperation(builder, internalOp);
  //     scheduledOps.insert(internalOp);
  //   }

  //   // Erase the scheduled bucket
  //   buckets.erase(minIt);
  // }

  // Compute scheduling order
  std::vector<Bucket> orderedBuckets;
  llvm::DenseSet<Operation *> alreadyProduced;
  while (!buckets.empty()) {
    auto &next = buckets.back();
    // Check all producerers are already in the orderedBuckets
    if (llvm::all_of(next.producerOps, [&alreadyProduced](Operation *op) {
          return alreadyProduced.count(op) != 0;
        })) {
      orderedBuckets.push_back(next);
      alreadyProduced.insert(next.rootOp);
      buckets.pop_back();
    } else {
      // Schedule necessary producer
      auto rIt = llvm::find_if(next.producerOps, [&](Operation *op) {
        return alreadyProduced.count(op) == 0;
      });
      assert(rIt != next.producerOps.end() && "expected to find unsatisfied dependency");
      auto bIt = llvm::find_if(
          buckets, [&](Bucket &bucket) { return bucket.rootOp == *rIt; });
      assert(bIt != buckets.end() && "expected to find producer bucket");
      auto bucket = *bIt;
      buckets.erase(bIt);
      buckets.push_back(bucket);
    }
  }

  for (auto &bucket : orderedBuckets) {
    // Schedule the actual bucket
    for (auto accessOp : bucket.accessOps) {
      if (scheduledOps.count(accessOp) == 0) {
        auto clonedOp = cloneOperation(builder, accessOp);
        scheduledOps.insert(accessOp);
      }
    }
    for (auto internalOp : bucket.internalOps) {
      auto clonedOp = cloneOperation(builder, internalOp);
      scheduledOps.insert(internalOp);
    }
  }

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
  funcPm.addPass(createStencilShufflePass());
  funcPm.addPass(createStencilPostShufflePass());
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