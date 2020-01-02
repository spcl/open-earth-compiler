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

// Helper method to shift a bound
SmallVector<int64_t, 3> shiftOffset(ArrayRef<int64_t> bound,
                                    ArrayRef<int64_t> shift) {
  assert(bound.size() == shift.size() &&
         "expected bound and shift to have the same size");
  SmallVector<int64_t, 3> result(bound.size());
  llvm::transform(llvm::zip(bound, shift), result.begin(),
                  [](std::tuple<int64_t, int64_t> x) {
                    return std::get<0>(x) - std::get<1>(x);
                  });
  return result;
}

struct ShapeShiftPass : public FunctionPass<ShapeShiftPass> {
  void runOnFunction() override;
};

} // namespace

void ShapeShiftPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!stencil::StencilDialect::isStencilProgram(funcOp))
    return;

  // Adapt the access offsets to the positive range
  funcOp.walk([](stencil::ApplyOp applyOp) {
    // Get the output bound
    ArrayRef<int64_t> output = applyOp.getLB();
    // Get the input bound of the operand
    for (unsigned i = 0, e = applyOp.getNumOperands(); i != e; ++i) {
      SmallVector<int64_t, 3> input;
      // Compute the shift for the operand
      auto operand = applyOp.getOperand(i);
      if (auto loadOp = dyn_cast<stencil::LoadOp>(operand.getDefiningOp())) {
        input = loadOp.getLB();
      }
      if (auto applyOp = dyn_cast<stencil::ApplyOp>(operand.getDefiningOp())) {
        input = applyOp.getLB();
      }
      // Shift all accesses of the corresponding operand
      auto argument = applyOp.getBody()->getArgument(i);
      applyOp.walk([&](stencil::AccessOp accessOp) {
        if (accessOp.view() == argument)
          accessOp.setOffset(
              shiftOffset(accessOp.getOffset(), shiftOffset(input, output)));
      });
    }
  });

  // Adapt the loop bounds of all apply ops to start start at zero
  funcOp.walk([](stencil::ApplyOp applyOp) {
    ArrayRef<int64_t> shift = applyOp.getLB();
    applyOp.setLB(shiftOffset(applyOp.getLB(), shift));
    applyOp.setUB(shiftOffset(applyOp.getUB(), shift));
  });

  // Adapt the bounds for all loads and stores
  funcOp.walk([](stencil::AssertOp assertOp) {
    // Adapt bounds by the lower bound of the assert op
    ArrayRef<int64_t> shift = assertOp.getLB();
    for (auto &use : assertOp.field().getUses()) {
      if (auto loadOp = dyn_cast<stencil::LoadOp>(use.getOwner())) {
        loadOp.setLB(shiftOffset(loadOp.getLB(), shift));
        loadOp.setUB(shiftOffset(loadOp.getUB(), shift));
      }
      if (auto storeOp = dyn_cast<stencil::StoreOp>(use.getOwner())) {
        storeOp.setLB(shiftOffset(storeOp.getLB(), shift));
        storeOp.setUB(shiftOffset(storeOp.getUB(), shift));
      }
    }

    // Adapt the assert bounds
    assertOp.setLB(shiftOffset(assertOp.getLB(), shift));
    assertOp.setUB(shiftOffset(assertOp.getUB(), shift));
  });
}

static PassRegistration<ShapeShiftPass>
    pass("stencil-shape-shift", "Shift the bounds to start at zero");
