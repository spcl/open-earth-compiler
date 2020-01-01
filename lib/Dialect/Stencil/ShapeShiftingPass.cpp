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
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include <bits/stdint-intn.h>
#include <cstddef>
#include <limits>

using namespace mlir;
using namespace stencil;

namespace {

struct ShapeShiftingPass : public FunctionPass<ShapeShiftingPass> {
  void runOnFunction() override;
};

} // namespace

void ShapeShiftingPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!stencil::StencilDialect::isStencilProgram(funcOp))
    return;

  // Adapt the access offsets in all apply ops
  funcOp.walk([](stencil::ApplyOp applyOp) {
    for (unsigned i = 0, e = applyOp.getNumOperands(); i != e; ++i) {
      auto operand = applyOp.getOperand(i);
      SmallVector<int64_t, 3> definingOpLB;
      // Compute the shift for the operand
      if (auto loadOp = dyn_cast<stencil::LoadOp>(operand.getDefiningOp())) {
        assert(loadOp.lb().hasValue() &&
               "expected load op to have valid bounds");
        definingOpLB = loadOp.getLB();
      }
      if (auto applyOp = dyn_cast<stencil::ApplyOp>(operand.getDefiningOp())) {
        assert(applyOp.lb().hasValue() &&
               "expected load op to have valid bounds");
        definingOpLB = applyOp.getLB();
      }
      // Shift all accesses of the corresponding operand
      auto argument = applyOp.getBody()->getArgument(i);
      applyOp.walk([&](stencil::AccessOp accessOp) {
        if (accessOp.view() == argument) {
          SmallVector<int64_t, 3> offset = accessOp.getOffset();
          llvm::transform(
              llvm::zip(offset, definingOpLB, applyOp.getLB()), offset.begin(),
              [](std::tuple<int64_t, int64_t, int64_t> x) {
                return std::get<0>(x) - (std::get<1>(x) - std::get<2>(x));
              });
          accessOp.setOffset(offset);
        }
      });
    }
  });

  // Adapt the loop bounds of all apply ops to start start at zero
  funcOp.walk([](stencil::ApplyOp applyOp) {
    assert(applyOp.lb().hasValue() && applyOp.ub().hasValue() &&
           "expected load op to have valid bounds");
    SmallVector<int64_t, 3> lb(applyOp.getLB().size(), 0);
    SmallVector<int64_t, 3> ub = applyOp.getUB();
    // Subtract the lower bound from the upper bound
    llvm::transform(llvm::zip(applyOp.getUB(), applyOp.getLB()), ub.begin(),
                    [](std::tuple<int64_t, int64_t> x) {
                      return std::get<0>(x) - std::get<1>(x);
                    });
    applyOp.setLB(lb);
    applyOp.setUB(ub);
  });
}

static PassRegistration<ShapeShiftingPass>
    pass("stencil-shape-shifting",
         "Shift the stencil apply bounds to make all access offsets positive.");
