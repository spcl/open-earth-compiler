#include "Conversion/LoopsToCUDA/Passes.h"
#include "PassDetail.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
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
  rewriter.replaceOp(op, extOp.getResult());
}

struct AddRewrite : public OpRewritePattern<LLVM::AddOp> {
  AddRewrite(MLIRContext *context)
      : OpRewritePattern<LLVM::AddOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(LLVM::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    // Replace the add op if all operands are sign extended
    if (allOperandsAreSignExtended(addOp.getOperands())) {
      replaceArithmeticOperation<LLVM::AddOp>(addOp, rewriter);
      return success();
    }
    return failure();
  }
};

struct MulRewrite : public OpRewritePattern<LLVM::MulOp> {
  MulRewrite(MLIRContext *context)
      : OpRewritePattern<LLVM::MulOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(LLVM::MulOp mulOp,
                                PatternRewriter &rewriter) const override {
    // Replace the add op if all operands are sign extended
    if (allOperandsAreSignExtended(mulOp.getOperands())) {
      replaceArithmeticOperation<LLVM::MulOp>(mulOp, rewriter);
      return success();
    }
    return failure();
  }
};

struct CmpRewrite : public OpRewritePattern<LLVM::ICmpOp> {
  CmpRewrite(MLIRContext *context)
      : OpRewritePattern<LLVM::ICmpOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(LLVM::ICmpOp cmpOp,
                                PatternRewriter &rewriter) const override {
    // Replace the add op if all operands are sign extended
    if (allOperandsAreSignExtended(cmpOp.getOperands())) {
      auto newOp = rewriter.create<LLVM::ICmpOp>(
          cmpOp.getLoc(), cmpOp.predicate(),
          cmpOp.getOperand(0).getDefiningOp()->getOperand(0),
          cmpOp.getOperand(1).getDefiningOp()->getOperand(0));
      rewriter.replaceOp(cmpOp, newOp.getResult());
      return success();
    }
    return failure();
  }
};

struct ConstantRewrite : public OpRewritePattern<LLVM::ConstantOp> {
  ConstantRewrite(MLIRContext *context)
      : OpRewritePattern<LLVM::ConstantOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(LLVM::ConstantOp constOp,
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
        rewriter.replaceOp(constOp, extOp.getResult());
        return success();
      }
    }
    return failure();
  }
};

struct StencilIndexOptimizationPass
    : public StencilIndexOptimizationPassBase<StencilIndexOptimizationPass> {
  void runOnOperation() override;
};

void StencilIndexOptimizationPass::runOnOperation() {
  auto funcOp = getOperation();
  if (funcOp.getAttrOfType<UnitAttr>(
          gpu::GPUDialect::getKernelFuncAttrName())) {
    OwningRewritePatternList patterns;
    patterns.insert<AddRewrite, MulRewrite, CmpRewrite, ConstantRewrite>(
        &getContext());
    applyPatternsAndFoldGreedily(funcOp, patterns);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createStencilIndexOptimizationPass() {
  return std::make_unique<StencilIndexOptimizationPass>();
}
