#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>

using namespace mlir;
using namespace stencil;

namespace {

struct StencilDomainSplitPass
    : public StencilDomainSplitPassBase<StencilDomainSplitPass> {

  void runOnFunction() override;
};

void StencilDomainSplitPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!StencilDialect::isStencilProgram(funcOp))
    return;

  // Check shape inference has been executed
  bool hasShapeOpWithoutShape = false;
  funcOp.walk([&](stencil::ShapeOp shapeOp) {
    if (!shapeOp.hasShape())
      hasShapeOpWithoutShape = true;
  });
  if (!hasShapeOpWithoutShape) {
    funcOp.emitOpError("execute combine split before shape inference");
    signalPassFailure();
    return;
  }

  // Maps to remember domains associated with every operation
  llvm::DenseMap<Operation *, Index> negative;
  llvm::DenseMap<Operation *, Index> positive;

  // Go through the operations in reverse order
  Block &entryBlock = funcOp.getOperation()->getRegion(0).front();
  for (auto op = entryBlock.rbegin(); op != entryBlock.rend(); ++op) {
    // For stores enter their domain into global domain map
    if (stencil::StoreOp storeOp = dyn_cast<stencil::StoreOp>(*op)) {
      negative[storeOp.getOperation()] =
          cast<ShapeOp>(storeOp.getOperation()).getLB();
      positive[storeOp.getOperation()] =
          cast<ShapeOp>(storeOp.getOperation()).getUB();
      // For applies split them if they participate in multiple domains
    } else if (ApplyOp applyOp = dyn_cast<ApplyOp>(*op)) {
      // List of all domains an apply participates with
      SmallVector<std::tuple<Index, Index>, 10> domains;
      // Map of what use of what result of the apply participates in what domain
      DenseMap<OpOperand *, std::tuple<Index, Index>> useToDomain;
      // Loop over all results and all uses for every result
      for (auto result : applyOp.getOperation()->getResults()) {
        for (auto &use : result.getUses()) {
          // Check if domain of this use of this result already exists
          if (llvm::all_of(domains, [&](std::tuple<Index, Index> tuple) {
                return get<0>(tuple) != negative[use.getOwner()] ||
                       get<1>(tuple) != positive[use.getOwner()];
              })) {
            // If the domain does not already exist, add the new domain
            domains.push_back(std::tuple<Index, Index>(
                negative[use.getOwner()], positive[use.getOwner()]));
            useToDomain[&use] = std::tuple<Index, Index>(
                negative[use.getOwner()], positive[use.getOwner()]);
          } else {
            // If the domain already exists add the use to the domain
            for (auto domain : domains) {
              if (std::get<0>(domain) == negative[use.getOwner()] &&
                  std::get<1>(domain) == positive[use.getOwner()]) {
                useToDomain[&use] = std::tuple<Index, Index>(
                    negative[use.getOwner()], positive[use.getOwner()]);
              }
            }
          }
        }
      }

      // Set domain for applyOp in global map
      negative[applyOp.getOperation()] = std::get<0>(domains[0]);
      positive[applyOp.getOperation()] = std::get<1>(domains[0]);

      OpBuilder builder(applyOp);
      builder.setInsertionPointAfter(applyOp);

      // For every domain construct one copy of the applyOp
      for (size_t i = 1, e = domains.size(); i < e; i++) {
        Operation *clonedOp = builder.clone(*applyOp.getOperation());
        // Add domain of new applyOp to global map
        negative[clonedOp] = std::get<0>(domains[i]);
        positive[clonedOp] = std::get<1>(domains[i]);
        // Replace uses of results with the applyOp of the according domain
        for (auto res : llvm::enumerate(
                 llvm::zip(applyOp.getResults(), clonedOp->getResults()))) {
          std::get<0>(res.value())
              .replaceUsesWithIf(std::get<1>(res.value()),
                                 [&](OpOperand &operand) {
                                   return std::get<0>(useToDomain[&operand]) ==
                                              std::get<0>(domains[i]) &&
                                          std::get<1>(useToDomain[&operand]) ==
                                              std::get<1>(domains[i]);
                                 });
        }
      }
    }
  }
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createStencilDomainSplitPass() {
  return std::make_unique<StencilDomainSplitPass>();
}
