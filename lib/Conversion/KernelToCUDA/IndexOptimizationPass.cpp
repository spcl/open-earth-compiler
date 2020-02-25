#include "Conversion/KernelToCUDA/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
#include "llvm/Support/raw_ostream.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <iterator>
#include <limits>

using namespace mlir;

namespace {

/// Helper method that checks all operands have been sign extended
bool allOperandsAreSignExtended(OperandRange operands) {
  return llvm::all_of(operands, [](Value value) {
    if (auto extOp = dyn_cast<LLVM::SExtOp>(value.getDefiningOp())) {
      auto *llvmDialect =
          value.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
      return extOp.arg().getType() == LLVM::LLVMType::getInt32Ty(llvmDialect) &&
             extOp.res().getType() == LLVM::LLVMType::getInt64Ty(llvmDialect);
    }
    return false;
  });
}

// Helper method to replace the arithmetic operation
template <typename T>
void replaceArithmeticOperation(T op, PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto *llvmDialect =
      rewriter.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
  auto newOp =
      rewriter.create<T>(loc, LLVM::LLVMType::getInt32Ty(llvmDialect),
                         op.getOperand(0).getDefiningOp()->getOperand(0),
                         op.getOperand(1).getDefiningOp()->getOperand(0));
  auto extOp = rewriter.create<LLVM::SExtOp>(
      loc, LLVM::LLVMType::getInt64Ty(llvmDialect), newOp.getResult());
  op.getResult().replaceAllUsesWith(extOp.getResult());
  op.erase();
}

struct AddRewrite : public OpRewritePattern<LLVM::AddOp> {
  AddRewrite(MLIRContext *context)
      : OpRewritePattern<LLVM::AddOp>(context, /*benefit=*/1) {}

  PatternMatchResult matchAndRewrite(LLVM::AddOp addOp,
                                     PatternRewriter &rewriter) const override {
    // Replace the add op if all operands are sign extended
    if (allOperandsAreSignExtended(addOp.getOperands())) {
      replaceArithmeticOperation<LLVM::AddOp>(addOp, rewriter);
      return matchSuccess();
    }
    return matchFailure();
  }
};

struct MulRewrite : public OpRewritePattern<LLVM::MulOp> {
  MulRewrite(MLIRContext *context)
      : OpRewritePattern<LLVM::MulOp>(context, /*benefit=*/1) {}

  PatternMatchResult matchAndRewrite(LLVM::MulOp mulOp,
                                     PatternRewriter &rewriter) const override {
    // Replace the add op if all operands are sign extended
    if (allOperandsAreSignExtended(mulOp.getOperands())) {
      replaceArithmeticOperation<LLVM::MulOp>(mulOp, rewriter);
      return matchSuccess();
    }
    return matchFailure();
  }
};

struct ConstantRewrite : public OpRewritePattern<LLVM::ConstantOp> {
  ConstantRewrite(MLIRContext *context)
      : OpRewritePattern<LLVM::ConstantOp>(context, /*benefit=*/1) {}

  PatternMatchResult matchAndRewrite(LLVM::ConstantOp constOp,
                                     PatternRewriter &rewriter) const override {
    // convert all index types to 32-bit integer constants
    if (constOp.value().getType().isIndex()) {
      int64_t value = constOp.value().cast<IntegerAttr>().getInt();
      // Convert index constants if they are in the 32-bit integer range
      if (value < std::numeric_limits<int32_t>::max() &&
          value > std::numeric_limits<int32_t>::min()) {
        auto loc = constOp.getLoc();
        auto *llvmDialect =
            constOp.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
        auto newOp = rewriter.create<LLVM::ConstantOp>(
            loc, LLVM::LLVMType::getInt32Ty(llvmDialect),
            rewriter.getI32IntegerAttr(value));
        auto extOp = rewriter.create<LLVM::SExtOp>(
            loc, LLVM::LLVMType::getInt64Ty(llvmDialect), newOp.getResult());
        constOp.getResult().replaceAllUsesWith(extOp.getResult());
        constOp.erase();
        return matchSuccess();
      }
    }
    return matchFailure();
  }
};

struct IndexOptimizationPass
    : public OperationPass<IndexOptimizationPass, LLVM::LLVMFuncOp> {
  void runOnOperation() override;
};

void IndexOptimizationPass::runOnOperation() {
  auto funcOp = getOperation();
  if (funcOp.getAttrOfType<UnitAttr>(
          gpu::GPUDialect::getKernelFuncAttrName())) {
    OwningRewritePatternList patterns;
    patterns.insert<AddRewrite, MulRewrite, ConstantRewrite>(&getContext());
    applyPatternsGreedily(funcOp, patterns);
  }
}

} // namespace

std::unique_ptr<OpPassBase<LLVM::LLVMFuncOp>>
mlir::stencil::createIndexOptimizationPass() {
  return std::make_unique<IndexOptimizationPass>();
}

static PassRegistration<IndexOptimizationPass>
    pass("index-optimization", "Perform 32-bit index computation");
