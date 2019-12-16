#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
//#include "mlir/Conversion/StenToStandard/ConvertStenToStandard.h"

using namespace mlir;

namespace {

void inlineStencilFunctionCall(stencil::CallOp callOp) {}

struct StencilInliningPass : public FunctionPass<StencilInliningPass> {
  void runOnFunction() override;
};

void StencilInliningPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil functions: this is the only place
  // where a sten.call operation can appear
  if (!stencil::StencilDialect::isStencilFunction(funcOp))
    return;

  funcOp.walk([](stencil::CallOp callOp) { inlineStencilFunctionCall(callOp); });
}

} // namespace

static PassRegistration<StencilInliningPass> pass("stencil-inlining",
                                                  "Inline stencil functions");
