#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <functional>
#include <limits>

using namespace mlir;
using namespace stencil;

namespace {

struct ShapeShiftPass : public ShapeShiftPassBase<ShapeShiftPass> {
  void runOnFunction() override;
};

} // namespace

void ShapeShiftPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!stencil::StencilDialect::isStencilProgram(funcOp))
    return;

  // Collect the minimal lower
  Index shapeShift;
  funcOp.walk([&](stencil::AssertOp assertOp) {
    auto lb = cast<ShapeOp>(assertOp.getOperation()).getLB();
    if (shapeShift.empty()) {
      shapeShift = lb;
    } else {
      shapeShift = applyFunElementWise(shapeShift, lb, min);
    }
  });

  // Shift all shapes to the positive domain
  funcOp.walk([&](ShapeOp shapeOp) {
    // Verify all operations have a valid shape
    if (!shapeOp.hasShape()) {
      shapeOp.emitOpError("expected op to have shape");
      signalPassFailure();
      return;
    }
    // Update the shape
    shapeOp.setLB(applyFunElementWise(shapeOp.getLB(), shapeShift, std::minus<int64_t>()));
    shapeOp.setUB(applyFunElementWise(shapeOp.getUB(), shapeShift, std::minus<int64_t>()));
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createShapeShiftPass() {
  return std::make_unique<ShapeShiftPass>();
}