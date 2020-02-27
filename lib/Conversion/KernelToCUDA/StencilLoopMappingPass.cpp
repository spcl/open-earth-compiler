#include "Conversion/KernelToCUDA/Passes.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
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
void setTheGPUMappingAttributes(OpBuilder &b, loop::ParallelOp op,
                                int64_t mappingOffset) {
  SmallVector<Attribute, 3> attrs;
  attrs.reserve(op.getNumInductionVars());
  for (int i = 0, e = op.getNumInductionVars(); i < e; ++i) {
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
  op.setAttr(gpu::kMappingAttributeName, ArrayAttr::get(attrs, b.getContext()));
}

// Method tiling and mapping a parallel loop for the GPU execution
void tileAndMapParallelLoop(loop::ParallelOp op, ArrayRef<int64_t> blockSizes) {
  assert(op.getNumInductionVars() == 3 &&
         "expected three-dimensional parallel loops");
  assert(llvm::all_of(op.lowerBound(),
                      [](Value val) {
                        return verifyIsConstant(val) &&
                               getConstantValue(val) == 0;
                      }) &&
         "expected zero lower bounds");

  OpBuilder b(op);
  SmallVector<Value, 3> blockSizeConstants;
  blockSizeConstants.reserve(op.upperBound().size());
  for (size_t i = 0, end = op.upperBound().size(); i != end; ++i) {
    if (i < blockSizes.size())
      blockSizeConstants.push_back(
          b.create<ConstantIndexOp>(op.getLoc(), blockSizes[i]));
    else
      // Just pick 1 for the remaining dimensions.
      blockSizeConstants.push_back(b.create<ConstantIndexOp>(op.getLoc(), 1));
  }

  // Create a parallel loop over blocks and threads.

  // TODO directly compute the values!!
  SmallVector<Value, 3> newSteps;
  for (size_t i = 0, end = op.step().size(); i != end; ++i) {
    assert(verifyIsConstant(op.step()[i]) && "expected constant step");
    newSteps.push_back(b.create<ConstantIndexOp>(
        op.getLoc(), blockSizes[i] * getConstantValue(op.step()[i])));
  }
  auto outerLoop = b.create<loop::ParallelOp>(op.getLoc(), op.lowerBound(),
                                              op.upperBound(), newSteps);
  b.setInsertionPointToStart(outerLoop.getBody());

  // Create the inner loop iterating over the block
  auto innerLoop = b.create<loop::ParallelOp>(op.getLoc(), op.lowerBound(),
                                              newSteps, op.step());
  // Steal the body of the old parallel loop and erase it.
  innerLoop.region().takeBody(op.region());

  // Add the loop indexes of both loops
  b.setInsertionPointToStart(innerLoop.getBody());
  auto expr = b.getAffineDimExpr(0) + b.getAffineDimExpr(1);
  auto map = AffineMap::get(2, 0, expr);
  // Replace all uses of loop induction variables with different constants
  SmallVector<ConstantIndexOp, 3> constants;
  for (auto inductionVar : innerLoop.getInductionVars()) {
    auto constantOp = b.create<ConstantIndexOp>(op.getLoc(), -1);
    inductionVar.replaceAllUsesWith(constantOp.getResult());
    constants.push_back(constantOp);
  }
  // Compute the actual induction variable
  for (size_t i = 0, end = op.upperBound().size(); i != end; ++i) {
    // TODO avoid addition if not needed
    ValueRange params = {*(outerLoop.getInductionVars().begin() + i),
                         *(innerLoop.getInductionVars().begin() + i)};
    auto affineApplyOp = b.create<AffineApplyOp>(op.getLoc(), map, params);
    constants[i].getResult().replaceAllUsesWith(affineApplyOp.getResult());
    constants[i].erase();
  }
  // Set the loop mapping for the different loop nests
  setTheGPUMappingAttributes(b, outerLoop, 0);
  setTheGPUMappingAttributes(b, innerLoop, outerLoop.lowerBound().size());

  op.erase();
}

struct StencilLoopMappingPass : public FunctionPass<StencilLoopMappingPass> {
  StencilLoopMappingPass() = default;
  StencilLoopMappingPass(const StencilLoopMappingPass &) {
  } // blockSize is non-copyable.

  void runOnFunction() override;

  ListOption<int64_t> blockSizes{
      *this, "stencil-loop-mapping-block-sizes",
      llvm::cl::desc("block sizes used for the mapping"), llvm::cl::ZeroOrMore,
      llvm::cl::MiscFlags::CommaSeparated};
};

void StencilLoopMappingPass::runOnFunction() {
  auto funcOp = getOperation();
  funcOp.walk(
      [&](loop::ParallelOp op) { tileAndMapParallelLoop(op, blockSizes); });
}

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> stencil::createStencilLoopMappingPass() {
  return std::make_unique<StencilLoopMappingPass>();
}

static PassRegistration<StencilLoopMappingPass>
    pass("stencil-loop-mapping", "Map parallel loops to blocks and threads");
