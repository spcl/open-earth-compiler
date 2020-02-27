#include "Conversion/KernelToCUDA/Passes.h"
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
                                int64_t mappingOffset) {
  SmallVector<Attribute, 3> attrs;
  attrs.reserve(parallelOp.getNumInductionVars());
  for (int i = 0, e = parallelOp.getNumInductionVars(); i < e; ++i) {
    // Map the last loop next to threads
    SmallVector<NamedAttribute, 3> entries;
    entries.emplace_back(b.getNamedAttr(
        gpu::kProcessorEntryName, b.getI64IntegerAttr(i + mappingOffset)));
    entries.emplace_back(b.getNamedAttr(
        gpu::kIndexMapEntryName, AffineMapAttr::get(b.getDimIdentityMap())));
    entries.emplace_back(b.getNamedAttr(
        gpu::kBoundMapEntryName, AffineMapAttr::get(b.getDimIdentityMap())));
    attrs.push_back(DictionaryAttr::get(entries, b.getContext()));
  }
  parallelOp.setAttr(gpu::kMappingAttributeName,
                     ArrayAttr::get(attrs, b.getContext()));
}

// Method tiling and mapping a parallel loop for the GPU execution
void tileAndMapParallelLoop(loop::ParallelOp parallelOp,
                            ArrayRef<int64_t> blockSizes) {
  assert(parallelOp.getNumInductionVars() == 3 &&
         "expected three-dimensional parallel loops");
  assert(llvm::all_of(parallelOp.lowerBound(),
                      [](Value val) {
                        return verifyIsConstant(val) &&
                               getConstantValue(val) == 0;
                      }) &&
         "expected zero lower bounds");

  OpBuilder b(parallelOp);
  SmallVector<Value, 3> blockSizeConstants;
  blockSizeConstants.reserve(parallelOp.upperBound().size());
  for (size_t i = 0, end = parallelOp.upperBound().size(); i != end; ++i) {
    if (i < blockSizes.size())
      blockSizeConstants.push_back(
          b.create<ConstantIndexOp>(parallelOp.getLoc(), blockSizes[i]));
    else
      // Just pick 1 for the remaining dimensions.
      blockSizeConstants.push_back(
          b.create<ConstantIndexOp>(parallelOp.getLoc(), 1));
  }

  // Create a parallel loop over blocks and threads.
  SmallVector<Value, 3> newSteps;
  for (size_t i = 0, end = parallelOp.step().size(); i != end; ++i) {
    assert(verifyIsConstant(parallelOp.step()[i]) && "expected constant step");
    newSteps.push_back(b.create<ConstantIndexOp>(
        parallelOp.getLoc(),
        blockSizes[i] * getConstantValue(parallelOp.step()[i])));
  }
  auto outerLoop =
      b.create<loop::ParallelOp>(parallelOp.getLoc(), parallelOp.lowerBound(),
                                 parallelOp.upperBound(), newSteps);
  b.setInsertionPointToStart(outerLoop.getBody());

  // Create the inner loop iterating over the block
  auto innerLoop =
      b.create<loop::ParallelOp>(parallelOp.getLoc(), parallelOp.lowerBound(),
                                 newSteps, parallelOp.step());

  // // Add a guard for out-of-bounds execution if needed
  // if (llvm::any_of(llvm::zip(op.upperBound(), blockSizeConstants),
  //                  [](std::tuple<int64_t, int64_t> x) {
  //                    return std::get<0>(x) % std::get<1>(x) != 0;
  //                  })) {
  //   b.setInsertionPointToStart(outerLoop.getBody());
  //   // Add compare only the necessary dimensions
  //   for (size_t i = 0, end = op.upperBound().size(); i != end; ++i) {
  // }

  // Steal the body of the old parallel loop and erase it.
  // innerLoop.region().takeBody(op.region());

  // Add the loop indexes of both loops
  b.setInsertionPointToStart(innerLoop.getBody());
  auto expr = b.getAffineDimExpr(0) + b.getAffineDimExpr(1);
  auto map = AffineMap::get(2, 0, expr);
  SmallVector<Value, 3> loopIVs;
  // Compute the actual induction variable
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

  // Clone the loop body and map the new arguments
  BlockAndValueMapping mapper;
  for (size_t i = 0, e = parallelOp.getNumInductionVars(); i < e; ++i) {
    mapper.map(parallelOp.getBody()->getArgument(i), loopIVs[i]);
  }
  for (auto &currentOp : parallelOp.getBody()->getOperations()) {
    if(currentOp.isKnownNonTerminator())
      b.clone(currentOp, mapper);
  }
  
  // Set the loop mapping for the different loop nests
  setTheGPUMappingAttributes(b, outerLoop, 0);
  setTheGPUMappingAttributes(b, innerLoop, outerLoop.lowerBound().size());

  // Erase the original loop
  parallelOp.erase();
}

struct StencilLoopMappingPass : public FunctionPass<StencilLoopMappingPass> {
  StencilLoopMappingPass() = default;
  StencilLoopMappingPass(const StencilLoopMappingPass &) {
  } // blockSize is non-copyable.

  void runOnFunction() override;

  ListOption<int64_t> blockSizes{
      *this, "block-sizes", llvm::cl::desc("block sizes used for the mapping"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};

void StencilLoopMappingPass::runOnFunction() {
  auto funcOp = getOperation();
  funcOp.walk([&](loop::ParallelOp parallelOp) {
    tileAndMapParallelLoop(parallelOp, blockSizes);
  });
}

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> stencil::createStencilLoopMappingPass() {
  return std::make_unique<StencilLoopMappingPass>();
}

static PassRegistration<StencilLoopMappingPass>
    pass("stencil-loop-mapping", "Map parallel loops to blocks and threads");
