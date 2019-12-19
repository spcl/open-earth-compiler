#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace stencil;

namespace {

struct ShapeInferencePass : public FunctionPass<ShapeInferencePass> {
  void runOnFunction() override;
};

// todo add a shape analysis pass

int checkKnownShape(ArrayRef<int64_t> shape) {
  for (int i = 0, e = shape.size(); i < e; ++i) {
    if (shape[i] < e)
      return i;
  }
  return -1;
}

// bool inferShapes(stencil::LoadOp loadOp) {
//   // // The shape of the field must be kwown at this point
//   // stencil::FieldType fieldType = storeOp.getFieldType();
//   // if (int idx = checkKnownShape(fieldType.getShape()) >= 0) {
//   //   storeOp.emitError("field shape dimension #") << idx << " must be known";
//   //   return false;
//   // }

//   // // Propagate the field shape to the view
//   // Type elementType = storeOp.getViewType().getElementType();
//   // storeOp.view()->setType(stencil::ViewType::get(
//   //     storeOp.getContext(), elementType, fieldType.getShape()));

//   return true;
// }

bool inferShapes(stencil::ApplyOp applyOp) {
  auto ctx = applyOp.getContext();

  // TODO probably makes sense to inline into region before? 

  // Iterate all result
  // TODO deal with multiple resutls
  Value* Result = applyOp.getResult();
  for(OpOperand &use : Result->getUses()) {
    if (auto storeOp = dyn_cast<stencil::StoreOp>(use.getOwner())) {
      auto LB = storeOp.getLB();
      auto UB = storeOp.getUB();

      llvm::outs() << "["
        << LB[0] << ":" << UB[0] << ","
        << LB[1] << ":" << UB[1] << ","
        << LB[2] << ":" << UB[2] << "]\n";
    }
  }

  // // The shape of the field must be kwown at this point
  // stencil::ViewType resultViewType = applyOp.getResultViewType();
  // if (int idx = checkKnownShape(resultViewType.getShape()) >= 0) {
  //   applyOp.emitError("view shape dimension #") << idx << " must be known";
  //   return false;
  // }

  // struct Extents {
  //   int64_t iplus, iminus;
  //   int64_t jplus, jminus;
  //   int64_t kplus, kminus;
  // };
  // DenseMap<Value *, Extents> argumentToExtents;
  // DenseMap<Value *, Value *> operandToArgument;

  // FuncOp callee = applyOp.getCallee();
  // for (Value *arg : callee.getArguments()) {
  //   Type argType = arg->getType();
  //   if (argType.isa<stencil::ViewType>())
  //     argumentToExtents[arg] = {0, 0, 0, 0, 0, 0};
  // }
  // for (int i = 0, e = callee.getNumArguments(); i < e; ++i) {
  //   Value *operand = applyOp.getOperand(i);
  //   if (operand->getType().isa<stencil::ViewType>()) {
  //     operandToArgument[operand] = callee.getArgument(i);
  //   }
  // }

  // callee.walk([&](stencil::AccessOp accessOp) {
  //   auto offset = accessOp.getOffset();
  //   Value *view = accessOp.view();

  //   Extents extents = argumentToExtents[view];
  //   extents.iminus = std::min(extents.iminus, offset[0]);
  //   extents.iplus = std::max(extents.iplus, offset[0]);
  //   extents.jminus = std::min(extents.jminus, offset[1]);
  //   extents.jplus = std::max(extents.jplus, offset[1]);
  //   extents.kminus = std::min(extents.kminus, offset[2]);
  //   extents.kplus = std::max(extents.kplus, offset[2]);

  //   argumentToExtents[view] = extents;
  // });

  // SmallVector<Attribute, 3> offsetsArray;

  // ArrayRef<int64_t> shape = resultViewType.getShape();
  // for (auto operand : applyOp.getOperands()) {
  //   Extents extents = argumentToExtents[operandToArgument[operand]];
  //   int iextent = extents.iplus - extents.iminus;
  //   int jextent = extents.jplus - extents.jminus;
  //   int kextent = extents.kplus - extents.kminus;
  //   ArrayRef<int64_t> newShape = {shape[0] + iextent, shape[1] + jextent,
  //                                 shape[2] + kextent};
  //   stencil::ViewType viewType =
  //       stencil::ViewType::get(ctx, resultViewType.getElementType(), newShape);
  //   operand->setType(viewType);

  //   NamedAttributeList extentAttr;
  //   Type i64 = IntegerType::get(64, ctx);
  //   extentAttr.set(Identifier::get("ioffset", ctx),
  //                  IntegerAttr::get(i64, -extents.iminus));
  //   extentAttr.set(Identifier::get("joffset", ctx),
  //                  IntegerAttr::get(i64, -extents.jminus));
  //   extentAttr.set(Identifier::get("koffset", ctx),
  //                  IntegerAttr::get(i64, -extents.kminus));
  //   offsetsArray.push_back(extentAttr.getDictionary());
  // }

  // applyOp.setAttr("offsets", ArrayAttr::get(offsetsArray, ctx));

  return true;
}

} // namespace

void ShapeInferencePass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!stencil::StencilDialect::isStencilProgram(funcOp))
    return;

  // Go through the operations in reverse order
  Block &entryBlock = funcOp.getOperation()->getRegion(0).front();
  for (auto op = entryBlock.rbegin(); op != entryBlock.rend(); ++op) {
    if (auto applyOp = dyn_cast<stencil::ApplyOp>(*op)) {
      if (!inferShapes(applyOp))
        signalPassFailure();
    }
  }
}

static PassRegistration<ShapeInferencePass>
    pass("stencil-shape-inference", "Infer the shapes of views and fields.");
