#include "Conversion/KernelToCUDA/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>

using namespace mlir;

namespace {

struct ConstantRewrite : public OpRewritePattern<LLVM::ConstantOp> {
  ConstantRewrite(MLIRContext *context)
      : OpRewritePattern<LLVM::ConstantOp>(context, /*benefit=*/1) {}

  PatternMatchResult matchAndRewrite(LLVM::ConstantOp constOp,
                                     PatternRewriter &rewriter) const
                                     override {
    if(constOp.getType())
    
    return matchFailure();
  }
};

// struct PostponeRewrite : public OpRewritePattern<LLVM::AddOp> {
//   PostponeRewrite(MLIRContext *context)
//       : OpRewritePattern<stencil::ApplyOp>(context, /*benefit=*/1) {}

//   PatternMatchResult matchAndRewrite(stencil::ApplyOp applyOp,
//                                      PatternRewriter &rewriter) const
//                                      override {
//     // Search producer apply op
//     for (auto operand : applyOp.operands()) {
//       if (isa_and_nonnull<stencil::ApplyOp>(operand.getDefiningOp())) {
//         // Check if multiple consumers
//         auto producerResults = operand.getDefiningOp()->getResults();
//         for (auto result : producerResults) {
//           if (llvm::any_of(result.getUsers(), [&](Operation *op) {
//                 return op != applyOp.getOperation();
//               }))
//             return matchFailure();
//         }

//         // If there is only a single consumer perform the inlining
//         return
//         inlineProducer(cast<stencil::ApplyOp>(operand.getDefiningOp()),
//                               applyOp, producerResults, rewriter);
//       }
//     }
//     return matchFailure();
//   }
// };

struct IndexOptimizationPass
    : public OperationPass<IndexOptimizationPass, LLVM::LLVMFuncOp> {
  void runOnOperation() override;
};

void IndexOptimizationPass::runOnOperation() {
  auto funcOp = getOperation();
  if (funcOp.getAttrOfType<UnitAttr>(
          gpu::GPUDialect::getKernelFuncAttrName())) {

    // OwningRewritePatternList patterns;
    // patterns.insert<PostponeRewrite>(&getContext());
    // applyPatternsGreedily(funcOp, patterns);
  }
}

} // namespace

std::unique_ptr<OpPassBase<LLVM::LLVMFuncOp>>
mlir::createIndexOptimizationPass() {
  return std::make_unique<IndexOptimizationPass>();
}

static PassRegistration<IndexOptimizationPass>
    pass("index-optimization", "Perform 32-bit index computation");
