#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include <bits/stdint-intn.h>
#include <limits>

using namespace mlir;
using namespace stencil;

namespace {

// Apply binary function element by element
SmallVector<int64_t, 3>
map(const SmallVector<int64_t, 3> &x, const SmallVector<int64_t, 3> &y,
    const std::function<int64_t(int64_t, int64_t)> &fun) {
  assert(x.size() == y.size() && "expected both vectors have equal size");
  SmallVector<int64_t, 3> result;
  for (int i = 0, e = x.size(); i != e; ++i) {
    result[i] = fun(x[i], y[i]);
  }
  return result;
}

/// This class computes for every stencil apply operand
/// the minimal bounding box containing all access offsets.
class AccessExtents {
  // This struct stores the positive and negative extends
  struct Extent {
    SmallVector<int64_t, 3> positive;
    SmallVector<int64_t, 3> negative;
  };

public:
  AccessExtents(stencil::ApplyOp applyOp) {
    // Compute mapping between operands and block arguments
    llvm::DenseMap<Value *, Value *> argumentsToOperands;
    for (int i = 0, e = applyOp.operands().size(); i != e; ++i) {
      argumentsToOperands[applyOp.getBody()->getArgument(i)] =
          applyOp.operands()[i];
      extents[applyOp.operands()[i]] = {{0, 0, 0}, {0, 0, 0}};
    }
    // Walk the access ops and update the extent
    applyOp.walk([&](stencil::AccessOp accessOp) {
      auto offset = accessOp.getOffset();
      auto argument = accessOp.getOperand();
      auto &ext = extents[argumentsToOperands[argument]];
      ext.negative = map(ext.negative, offset,
                         [](int64_t x, int64_t y) { return std::min(x, y); });
      ext.positive = map(ext.positive, offset,
                         [](int64_t x, int64_t y) { return std::max(x, y); });
    });
  }

  Extent *lookupExtent(Value *value) {
    auto extent = extents.find(value);
    if (extent == extents.end())
      return nullptr;
    else
      return &extent->second;
  }

private:
  llvm::DenseMap<Value *, Extent> extents;
};

struct ShapeInferencePass : public FunctionPass<ShapeInferencePass> {
  void runOnFunction() override;
};

bool inferShapes(stencil::ApplyOp applyOp) {
  // Initial lower and upper bounds
  SmallVector<int64_t, 3> lowerBound = {std::numeric_limits<int64_t>::max(),
                                        std::numeric_limits<int64_t>::max(),
                                        std::numeric_limits<int64_t>::max()};
  SmallVector<int64_t, 3> upperBound = {std::numeric_limits<int64_t>::min(),
                                        std::numeric_limits<int64_t>::min(),
                                        std::numeric_limits<int64_t>::min()};
  // Iterate all uses and extend the bounds
  for (auto result : applyOp.getResults()) {
    for (OpOperand &use : result->getUses()) {
      if (auto storeOp = dyn_cast<stencil::StoreOp>(use.getOwner())) {
        lowerBound = map(lowerBound, storeOp.getLB(),
                         [](int64_t x, int64_t y) { return std::min(x, y); });
        upperBound = map(upperBound, storeOp.getUB(),
                         [](int64_t x, int64_t y) { return std::max(x, y); });
      }
      if (auto applyOp = dyn_cast<stencil::ApplyOp>(use.getOwner())) {
      }
    }
  }
  // applyOp.setAttr("lb", lowerBound);
  // applyOp.setAttr("ub", upperBound);

  // TODO probably makes sense to inline into region before?

  // Iterate all result
  // TODO deal with multiple resutls
  // Value* Result = applyOp.getResult();
  // for(OpOperand &use : Result->getUses()) {
  //   if (auto storeOp = dyn_cast<stencil::StoreOp>(use.getOwner())) {
  //     auto LB = storeOp.getLB();
  //     auto UB = storeOp.getUB();

  //     llvm::outs() << "["
  //       << LB[0] << ":" << UB[0] << ","
  //       << LB[1] << ":" << UB[1] << ","
  //       << LB[2] << ":" << UB[2] << "]\n";
  //   }
  // }

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
  //       stencil::ViewType::get(ctx, resultViewType.getElementType(),
  //       newShape);
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
