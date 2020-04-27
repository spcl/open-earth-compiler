#include "Conversion/LoopsToCUDA/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "PassDetail.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
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
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>
#include <limits>

using namespace mlir;

namespace {

enum MappingLevel { MapGrid = 0, MapBlock = 1, Sequential = 2 };

// Helper method to get the hardware id for the mapping
gpu::Processor getHardwareId(MappingLevel level, int dimension) {
  static constexpr int kNumHardwareIds = 3;
  if (dimension >= kNumHardwareIds || level == Sequential)
    return gpu::Processor::Sequential;
  switch (level) {
  case MapGrid:
    switch (dimension) {
    case 0:
      return gpu::Processor::BlockX;
    case 1:
      return gpu::Processor::BlockY;
    case 2:
      return gpu::Processor::BlockZ;
    default:
      return gpu::Processor::Sequential;
    }
    break;
  case MapBlock:
    switch (dimension) {
    case 0:
      return gpu::Processor::ThreadX;
    case 1:
      return gpu::Processor::ThreadY;
    case 2:
      return gpu::Processor::ThreadZ;
    default:
      return gpu::Processor::Sequential;
    }
  default:;
  }
  return gpu::Processor::Sequential;
}

// Helper method verifying the lower bounds are zero
bool verifyIsConstant(Value val) {
  return llvm::isa_and_nonnull<ConstantIndexOp>(val.getDefiningOp());
}

// Helper method returning a constant value
int64_t getConstantValue(Value val) {
  auto constantOp = cast<ConstantIndexOp>(val.getDefiningOp());
  return constantOp.getValue();
}

// Helper method setting the loop mapping attributes for a parallel loop
void setTheGPUMappingAttributes(OpBuilder &b, loop::ParallelOp parallelOp,
                                MappingLevel level) {
  // TODO fix this
  SmallVector<gpu::ParallelLoopDimMapping, 3> attrs;
  attrs.reserve(parallelOp.getNumLoops());
  for (int i = 0, e = parallelOp.getNumLoops(); i < e; ++i) {
    attrs.push_back(gpu::getParallelLoopDimMappingAttr(
        getHardwareId(level, i), b.getDimIdentityMap(), b.getDimIdentityMap()));
  }
  gpu::setMappingAttr(parallelOp, attrs);
}

// Method tiling and mapping a parallel loop for the GPU execution
void tileAndMapParallelLoop(loop::ParallelOp parallelOp,
                            ArrayRef<int64_t> blockSizes) {
  assert(parallelOp.getNumLoops() == stencil::kNumOfDimensions &&
         "expected parallel loop to have full dimensionality");
  assert(llvm::all_of(parallelOp.lowerBound(),
                      [](Value val) {
                        return verifyIsConstant(val) &&
                               getConstantValue(val) == 0;
                      }) &&
         "expected zero lower bounds");
  // Prepare the builder
  OpBuilder b(parallelOp);

  // Compute the block sizes assuming 1 if no config provided
  SmallVector<Value, 3> blockSizeConstants;
  SmallVector<int64_t, 3> loopBounds;
  for (size_t i = 0, end = parallelOp.upperBound().size(); i != end; ++i) {
    if (i < blockSizes.size())
      blockSizeConstants.push_back(
          b.create<ConstantIndexOp>(parallelOp.getLoc(), blockSizes[i]));
    else
      blockSizeConstants.push_back(
          b.create<ConstantIndexOp>(parallelOp.getLoc(), 1));
    loopBounds.push_back(getConstantValue(parallelOp.upperBound()[i]));
  }

  // Create a parallel loop over blocks and threads.
  SmallVector<Value, 3> newStepConstants;
  SmallVector<int64_t, 3> newSteps;
  SmallVector<Value, 3> newUpperBound;
  for (size_t i = 0, end = parallelOp.step().size(); i != end; ++i) {
    assert(verifyIsConstant(parallelOp.step()[i]) && "expected constant step");
    int64_t newStep = getConstantValue(blockSizeConstants[i]) *
                      getConstantValue(parallelOp.step()[i]);
    newStepConstants.push_back(
        b.create<ConstantIndexOp>(parallelOp.getLoc(), newStep));
    newSteps.push_back(newStep);
    newUpperBound.push_back(b.create<ConstantIndexOp>(
        parallelOp.getLoc(),
        (newStep * ((loopBounds[i] + newStep - 1) / newStep))));
  }
  auto outerLoop =
      b.create<loop::ParallelOp>(parallelOp.getLoc(), parallelOp.lowerBound(),
                                 newUpperBound, newStepConstants);
  b.setInsertionPointToStart(outerLoop.getBody());

  // Create the inner loop iterating over the block
  auto innerLoop =
      b.create<loop::ParallelOp>(parallelOp.getLoc(), parallelOp.lowerBound(),
                                 newStepConstants, parallelOp.step());

  // Sum the loop induction variables if necessary
  b.setInsertionPointToStart(innerLoop.getBody());
  auto expr = b.getAffineDimExpr(0) + b.getAffineDimExpr(1);
  auto map = AffineMap::get(2, 0, expr);
  SmallVector<Value, 3> loopIVs;
  for (size_t i = 0, end = parallelOp.upperBound().size(); i != end; ++i) {
    // Use the inner loop iv if the outer loop performs only one iteration
    if (getConstantValue(outerLoop.upperBound()[i]) ==
        getConstantValue(outerLoop.step()[i])) {
      loopIVs.push_back(*(innerLoop.getInductionVars().begin() + i));
      continue;
    }
    // Use the outer loop iv if the inner loop performs only one iteration
    if (getConstantValue(innerLoop.upperBound()[i]) ==
        getConstantValue(innerLoop.step()[i])) {
      loopIVs.push_back(*(outerLoop.getInductionVars().begin() + i));
      continue;
    }
    // Otherwise add the loop induction variables
    ValueRange params = {*(outerLoop.getInductionVars().begin() + i),
                         *(innerLoop.getInductionVars().begin() + i)};
    auto affineApplyOp =
        b.create<AffineApplyOp>(parallelOp.getLoc(), map, params);
    loopIVs.push_back(affineApplyOp.getResult());
  }

  // Add a guard for out-of-bounds execution if needed
  if (llvm::any_of(llvm::zip(loopBounds, newSteps),
                   [](std::tuple<int64_t, int64_t> x) {
                     return std::get<0>(x) % std::get<1>(x) != 0;
                   })) {
    // Add compare only the necessary dimensions
    SmallVector<Value, 3> predicates;
    for (size_t i = 0, end = parallelOp.upperBound().size(); i != end; ++i) {
      if (loopBounds[i] % newSteps[i] != 0) {
        auto cmpOp = b.create<CmpIOp>(parallelOp.getLoc(), CmpIPredicate::slt,
                                      loopIVs[i], parallelOp.upperBound()[i]);
        predicates.push_back(cmpOp);
      }
    }
    // Accumulate the conditions
    Value predicate = predicates.back();
    predicates.pop_back();
    while (!predicates.empty()) {
      predicate =
          b.create<AndOp>(parallelOp.getLoc(), predicates.back(), predicate);
      predicates.pop_back();
    }
    // Insert the guard
    auto ifOp = b.create<loop::IfOp>(parallelOp.getLoc(), predicate, false);
    b.setInsertionPointToStart(&ifOp.thenRegion().front());
  }

  // Clone the loop body and map the new arguments
  BlockAndValueMapping mapper;
  for (size_t i = 0, e = parallelOp.getNumLoops(); i < e; ++i) {
    mapper.map(parallelOp.getBody()->getArgument(i), loopIVs[i]);
  }
  for (auto &currentOp : parallelOp.getBody()->getOperations()) {
    if (currentOp.isKnownNonTerminator())
      b.clone(currentOp, mapper);
  }

  // Set the loop mapping for the different loop nests
  setTheGPUMappingAttributes(b, outerLoop, MapGrid);
  setTheGPUMappingAttributes(b, innerLoop, MapBlock);

  // Erase the original loop
  parallelOp.erase();
}

struct StencilLoopMappingPass
    : public StencilLoopMappingPassBase<StencilLoopMappingPass> {
  void runOnFunction() override;
};

void StencilLoopMappingPass::runOnFunction() {
  auto funcOp = getOperation();
  funcOp.walk([&](loop::ParallelOp parallelOp) {
    tileAndMapParallelLoop(parallelOp, blockSizes);
  });
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createStencilLoopMappingPass() {
  return std::make_unique<StencilLoopMappingPass>();
}
