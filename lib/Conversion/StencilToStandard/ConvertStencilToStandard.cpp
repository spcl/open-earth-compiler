#include "Conversion/StencilToStandard/ConvertStencilToStandard.h"
#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/raw_ostream.h"
//#include <bits/stdint-intn.h>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Rewriting Pattern
//===----------------------------------------------------------------------===//

// class FieldOpLowering : public ConversionPattern {
// public:
//   explicit FieldOpLowering(MLIRContext *context)
//       : ConversionPattern(stencil::FieldOp::getOperationName(), 1, context) {}

//   PatternMatchResult
//   matchAndRewrite(Operation *operation, ArrayRef<Value *> operands,
//                   ConversionPatternRewriter &rewriter) const override {

//     // Compute the memref type
//     stencil::FieldOp fieldOp = cast<stencil::FieldOp>(operation);
//     SmallVector<int64_t, 3> shape = {0, 0, 0};
//     llvm::transform(llvm::zip(fieldOp.getUB(), fieldOp.getLB()), shape.begin(),
//                     [](std::tuple<int64_t, int64_t> x) {
//                       return std::get<0>(x) - std::get<1>(x);
//                     });
//     Type elementType = fieldOp.getFieldType().getElementType();
//     MemRefType fieldType = MemRefType::get(shape, elementType);

    

//     // Insert
//     auto funcName = (Twine("get_") + Twine(fieldOp.name()) + Twine("_field")).str();
//     auto parentOp = operation->getParentOfType<FuncOp>();
//     rewriter.setInsertionPoint(parentOp);
//     auto funcType = rewriter.getFunctionType({}, {fieldType});
//     auto funcOp = rewriter.create<FuncOp>(operation->getLoc(), funcName, funcType, llvm::None);

    
//     rewriter.setInsertionPoint(operation);
//     auto callOp = rewriter.create<CallOp>(operation->getLoc(), funcOp);
//     operation->getResult(0)->replaceAllUsesWith(callOp.getResult(0));
    
//     rewriter.eraseOp(operation);

//     callOp.getParentOp()->getParentOp()->dump();


// //     auto newFuncType = rewriter.getFunctionType(inputs,
// //     funcType.getResults()); auto newFuncOp = rewriter.create<FuncOp>(
// //         operation->getLoc(),
// //         funcOp.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
// //             .getValue(),
// //         newFuncType, llvm::None);

//         // create a function declaration

//         // stencil::ViewType viewType =
//         //     cast<stencil::LoadOp>(operation).getResultViewType();
//         // rewriter.replaceOpWithNewOp<MemRefCastOp>(
//         //     operation, operands[0],
//         //     MemRefType::get(viewType.getShape(), viewType.getElementType()));

//         // MemRefType::get()

//     return matchSuccess();
//   }
// };

// class FuncOpLowering : public ConversionPattern {
// public:
//   explicit FuncOpLowering(MLIRContext *context)
//       : ConversionPattern(FuncOp::getOperationName(), 1, context) {}

//   PatternMatchResult
//   matchAndRewrite(Operation *operation, ArrayRef<Value *> operands,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto funcOp = cast<FuncOp>(operation);
//     auto funcType = funcOp.getType();

//     SmallVector<Type, 3> inputs;
//     for (const auto &type : funcType.getInputs()) {
//       if (auto fieldType = type.dyn_cast<stencil::FieldType>()) {
//         inputs.push_back(
//             MemRefType::get(fieldType.getShape(),
//             fieldType.getElementType()));
//       } else if (auto viewType = type.dyn_cast<stencil::ViewType>()) {
//         inputs.push_back(
//             MemRefType::get(viewType.getShape(), viewType.getElementType()));
//       } else
//         operation->emitError("unexpected argument type '") << type << "'";

//       // For stencil functions, add three parameters for each operand to pass
//       // the i, j and k coordinates of its origin
//       if (stencil::StencilDialect::isStencilFunction(funcOp)) {
//         for (int i = 0; i < 3; ++i)
//           inputs.push_back(IndexType::get(operation->getContext()));
//       }
//     }

//     auto newFuncType = rewriter.getFunctionType(inputs,
//     funcType.getResults()); auto newFuncOp = rewriter.create<FuncOp>(
//         operation->getLoc(),
//         funcOp.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
//             .getValue(),
//         newFuncType, llvm::None);

//     Block *entryBlock = newFuncOp.addEntryBlock();

//     for (int i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
//       if (stencil::StencilDialect::isStencilFunction(funcOp)) {
//         funcOp.getArgument(i)->replaceAllUsesWith(
//             entryBlock->getArgument(i * 4));
//       } else {
//         funcOp.getArgument(i)->replaceAllUsesWith(entryBlock->getArgument(i));
//       }
//     }
//     auto &operations =
//         funcOp.getOperation()->getRegion(0).front().getOperations();
//     entryBlock->getOperations().splice(entryBlock->begin(), operations);

//     rewriter.eraseOp(operation);

//     return matchSuccess();
//   }
// };

// class LoadOpLowering : public ConversionPattern {
// public:
//   explicit LoadOpLowering(MLIRContext *context)
//       : ConversionPattern(stencil::LoadOp::getOperationName(), 1, context) {}

//   PatternMatchResult
//   matchAndRewrite(Operation *operation, ArrayRef<Value *> operands,
//                   ConversionPatternRewriter &rewriter) const override {
//     stencil::ViewType viewType =
//         cast<stencil::LoadOp>(operation).getResultViewType();
//     rewriter.replaceOpWithNewOp<MemRefCastOp>(
//         operation, operands[0],
//         MemRefType::get(viewType.getShape(), viewType.getElementType()));

//     return matchSuccess();
//   }
// };

// class AccessOpLowering : public ConversionPattern {
// public:
//   explicit AccessOpLowering(MLIRContext *context)
//       : ConversionPattern(stencil::AccessOp::getOperationName(), 1, context)
//       {}

//   PatternMatchResult
//   matchAndRewrite(Operation *operation, ArrayRef<Value *> operands,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto loc = operation->getLoc();
//     auto accessOp = cast<stencil::AccessOp>(operation);

//     FuncOp funcOp = operation->getParentOfType<FuncOp>();
//     if (!funcOp) {
//       operation->emitError("expected parent function operation");
//       return matchFailure();
//     }

//     // Assume that the view is always a function argument
//     auto viewOperandIterator = llvm::find(funcOp.getArguments(),
//     operands[0]); size_t viewOperandPosition =
//         std::distance(funcOp.getArguments().begin(), viewOperandIterator);

//     // AccessOp can only appear in a stencil.function
//     Value *iVal = funcOp.getArgument(viewOperandPosition + 1);
//     Value *jVal = funcOp.getArgument(viewOperandPosition + 2);
//     Value *kVal = funcOp.getArgument(viewOperandPosition + 3);

//     ArrayRef<int64_t> offset = accessOp.getOffset();
//     auto iOffset = rewriter.create<ConstantIndexOp>(loc,
//     offset[0]).getResult(); auto jOffset =
//     rewriter.create<ConstantIndexOp>(loc, offset[1]).getResult(); auto
//     kOffset = rewriter.create<ConstantIndexOp>(loc, offset[2]).getResult();

//     AffineExpr addOffsetExpr =
//         rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
//     auto addOffsetMap = AffineMap::get(2, 0, addOffsetExpr);

//     auto i = rewriter
//                  .create<AffineApplyOp>(operation->getLoc(), addOffsetMap,
//                                         llvm::makeArrayRef({iVal, iOffset}))
//                  .getResult();
//     auto j = rewriter
//                  .create<AffineApplyOp>(operation->getLoc(), addOffsetMap,
//                                         llvm::makeArrayRef({jVal, jOffset}))
//                  .getResult();
//     auto k = rewriter
//                  .create<AffineApplyOp>(operation->getLoc(), addOffsetMap,
//                                         llvm::makeArrayRef({kVal, kOffset}))
//                  .getResult();

//     rewriter.replaceOpWithNewOp<LoadOp>(operation, operands[0],
//                                         llvm::makeArrayRef({i, j, k}));

//     return matchSuccess();
//   }
// };

// class ApplyOpLowering : public ConversionPattern {
// public:
//   explicit ApplyOpLowering(MLIRContext *context)
//       : ConversionPattern(stencil::ApplyOp::getOperationName(), 1, context)
//       {}

//   PatternMatchResult
//   matchAndRewrite(Operation *operation, ArrayRef<Value *> operands,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto loc = operation->getLoc();
//     auto applyOp = cast<stencil::ApplyOp>(operation);

//     // The apply operation should have an 'offsets' attribute that was
//     computed
//     // by the shape inference pass
//     auto offsetsAttr = applyOp.getAttrOfType<ArrayAttr>("offsets");
//     if (!offsetsAttr) {
//       operation->emitError("expected 'offsets' attribute");
//       return matchFailure();
//     }
//     SmallVector<DictionaryAttr, 3> offsets;
//     for (auto attr : offsetsAttr.getValue()) {
//       offsets.push_back(attr.cast<DictionaryAttr>());
//     }

//     // The loop bounds are given by the shape of the resulting view
//     stencil::ViewType resultViewType = applyOp.getResultViewType();
//     ArrayRef<int64_t> resultViewShape = resultViewType.getShape();

//     auto operandView =
//         rewriter
//             .create<AllocOp>(loc,
//                              MemRefType::get(resultViewShape,
//                                              resultViewType.getElementType()))
//             .getResult();

//     auto jLoop = rewriter.create<AffineForOp>(loc, 0, resultViewShape[1]);
//     rewriter.setInsertionPointToStart(jLoop.getBody());
//     auto iLoop = rewriter.create<AffineForOp>(loc, 0, resultViewShape[0]);
//     rewriter.setInsertionPointToStart(iLoop.getBody());
//     auto kLoop = rewriter.create<AffineForOp>(loc, 0, resultViewShape[2]);
//     rewriter.setInsertionPointToStart(kLoop.getBody());

//     AffineExpr addOffsetExpr =
//         rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1);
//     auto addOffsetMap = AffineMap::get(2, 0, addOffsetExpr);

//     SmallVector<Value *, 12> callOperands;
//     for (int operand = 0, e = applyOp.getNumOperands(); operand < e;
//          ++operand) {
//       // Cast away the static size
//       SmallVector<int64_t, 3> dynamicShape(resultViewType.getShape().size(),
//                                            -1);
//       auto dynamicSizeOperand =
//           rewriter
//               .create<MemRefCastOp>(
//                   loc, applyOp.getOperand(operand),
//                   MemRefType::get(dynamicShape,
//                                   resultViewType.getElementType()))
//               .getResult();
//       callOperands.push_back(dynamicSizeOperand);

//       DictionaryAttr operandOffsets = offsets[operand];
//       auto iOffsetAttr = operandOffsets.get("ioffset").cast<IntegerAttr>();
//       auto iOffset = rewriter.create<ConstantIndexOp>(loc,
//       iOffsetAttr.getInt())
//                          .getResult();
//       auto jOffsetAttr = operandOffsets.get("joffset").cast<IntegerAttr>();
//       auto jOffset = rewriter.create<ConstantIndexOp>(loc,
//       jOffsetAttr.getInt())
//                          .getResult();
//       auto kOffsetAttr = operandOffsets.get("koffset").cast<IntegerAttr>();
//       auto kOffset = rewriter.create<ConstantIndexOp>(loc,
//       kOffsetAttr.getInt())
//                          .getResult();

//       auto i = rewriter.create<AffineApplyOp>(
//           loc, addOffsetMap,
//           llvm::makeArrayRef({iLoop.getInductionVar(), iOffset}));
//       auto j = rewriter.create<AffineApplyOp>(
//           loc, addOffsetMap,
//           llvm::makeArrayRef({jLoop.getInductionVar(), jOffset}));
//       auto k = rewriter.create<AffineApplyOp>(
//           loc, addOffsetMap,
//           llvm::makeArrayRef({kLoop.getInductionVar(), kOffset}));

//       callOperands.push_back(i);
//       callOperands.push_back(j);
//       callOperands.push_back(k);
//     }

//     auto callOp =
//         rewriter.create<CallOp>(loc, applyOp.getCallee(), callOperands);

//     auto i = iLoop.getInductionVar();
//     auto j = jLoop.getInductionVar();
//     auto k = kLoop.getInductionVar();
//     rewriter.create<StoreOp>(loc, callOp.getResult(0), operandView,
//                              llvm::makeArrayRef({i, j, k}));

//     rewriter.replaceOp(operation, operandView);

//     return matchSuccess();
//   }
// };

// class StoreOpLowering : public ConversionPattern {
// public:
//   explicit StoreOpLowering(MLIRContext *context)
//       : ConversionPattern(stencil::StoreOp::getOperationName(), 1, context)
//       {}

//   PatternMatchResult
//   matchAndRewrite(Operation *operation, ArrayRef<Value *> operands,
//                   ConversionPatternRewriter &rewriter) const override {
//     /*auto loc = operation->getLoc();
//     auto storeOp = cast<stencil::StoreOp>(operation);
//     stencil::FieldType fieldType = storeOp.getFieldType();
//     ArrayRef<int64_t> shape = fieldType.getShape();

//     auto jLoop = rewriter.create<AffineForOp>(loc, 0, shape[1]);
//     rewriter.setInsertionPointToStart(jLoop.getBody());
//     auto iLoop = rewriter.create<AffineForOp>(loc, 0, shape[0]);
//     rewriter.setInsertionPointToStart(iLoop.getBody());
//     auto kLoop = rewriter.create<AffineForOp>(loc, 0, shape[2]);
//     rewriter.setInsertionPointToStart(kLoop.getBody());*/

//     // Replace all the uses of the source memref with the target memref
//     Value *target = operands[0];
//     Value *source = operands[1];
//     source->replaceAllUsesWith(target);
//     source->getDefiningOp()->erase();

//     rewriter.eraseOp(operation);

//     return matchSuccess();
//   }
// };

//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

class StencilToStandardTarget : public ConversionTarget {
public:
  explicit StencilToStandardTarget(MLIRContext &context)
      : ConversionTarget(context) {}

  // bool isDynamicallyLegal(Operation *op) const override {
  //   if (auto funcOp = dyn_cast<FuncOp>(op)) {
  //     return !funcOp.getAttr(
  //                stencil::StencilDialect::getStencilProgramAttrName()) &&
  //            !funcOp.getAttr(
  //                stencil::StencilDialect::getStencilProgramAttrName());
  //   } else
  //     return true;
  // }
};

//===----------------------------------------------------------------------===//
// Rewriting Pass
//===----------------------------------------------------------------------===//

struct StencilToStandardPass : public ModulePass<StencilToStandardPass> {
  void runOnModule() override;
};

void StencilToStandardPass::runOnModule() {
  OwningRewritePatternList patterns;
  auto module = getModule();

  populateStencilToStandardConversionPatterns(patterns, module.getContext());

  StencilToStandardTarget target(*(module.getContext()));
  target.addLegalDialect<AffineOpsDialect>();
  target.addLegalDialect<StandardOpsDialect>();

  //target.addLegalDialect<stencil::StencilDialect>();
  //target.addDynamicallyLegalOp<FuncOp>();

  if (failed(applyPartialConversion(module, target, patterns)))
    signalPassFailure();
}

} // namespace

void mlir::populateStencilToStandardConversionPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *ctx) {
  // patterns.insert<AccessOpLowering, ApplyOpLowering, FuncOpLowering,
  //                 LoadOpLowering, StoreOpLowering>(ctx);
  //patterns.insert<FieldOpLowering>(ctx);
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::stencil::createConvertStencilToStandardPass() {
  return std::make_unique<StencilToStandardPass>();
}

static PassRegistration<StencilToStandardPass>
    pass("convert-stencil-to-standard",
         "Convert stencil dialect to standard operations");
