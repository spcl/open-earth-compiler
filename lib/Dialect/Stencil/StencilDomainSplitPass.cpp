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
      // Order of dimension: NumResults, NumDomains, NumUses
      SmallVector<SmallVector<SmallVector<Operation *, 10>, 10>, 10> users;
      for(size_t k = 0; k < applyOp.getNumResults(); k++) {
        users.emplace_back(SmallVector<SmallVector<Operation*, 10>, 10>());
      }

      // Loop over all results and all uses for every result
      for(size_t k = 0; k < applyOp.getNumResults(); k++) {
        for (auto user : applyOp.getOperation()->getResult(k).getUsers()) {
          // Check if domain of this use of this result already exists
          if (llvm::all_of(domains, [&](std::tuple<Index, Index> tuple) {
                return get<0>(tuple) != negative[user] ||
                       get<1>(tuple) != positive[user];
              })) {
            // Add new domain, construct empty vectors for new domain
            domains.emplace_back(
                std::tuple<Index, Index>(negative[user], positive[user]));
            for(size_t k = 0; k < applyOp.getNumResults(); k++) {
              users[k].emplace_back(SmallVector<Operation *, 10>());
            }
            users[k].back().push_back(user);
          } else {
            // Add use to already existing domains
            for (size_t i = 0; i < domains.size(); i++) {
              if (std::get<0>(domains[i]) == negative[user] &&
                  std::get<1>(domains[i]) == positive[user]) {
                users[k][i].push_back(user);
              }
            }
          }
        }
      }

      // Set domain for applyOp in global datastructure
      negative[applyOp.getOperation()] = std::get<0>(domains[0]);
      positive[applyOp.getOperation()] = std::get<1>(domains[0]);

      // Construct blockList for substitution of k-th result
      SmallVector<SmallPtrSet<Operation*, 10>, 10> blockList;
      for(size_t k = 0; k < applyOp.getNumResults(); k++) {
        blockList.emplace_back(SmallPtrSet<Operation*, 10>());
        for (Operation *user : users[k][0]) {
          blockList[k].insert(user);
        }
      }

      OpBuilder builder(applyOp);
      builder.setInsertionPointAfter(applyOp);

      // For every domain construct one copy of the applyOp
      for (size_t i = 1; i < domains.size(); i++) {
        Operation *clonedOp = builder.clone(*applyOp.getOperation());
        // Add domain of new applyOp to global map
        negative[clonedOp] = std::get<0>(domains[i]);
        positive[clonedOp] = std::get<1>(domains[i]);
        // Replace all uses of all results of the applyOp for domains not yet
        // looked at
        for(size_t k = 0; k < applyOp.getNumResults(); k++) {
          applyOp.getResult(k).replaceAllUsesExcept(clonedOp->getResult(k),
                                                    blockList[k]);
          // Extend blockLists by current domain
          for (Operation *user : users[k][i]) {
            blockList[k].insert(user);
          }
        }
      }
    }
  }
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createStencilDomainSplitPass() {
  return std::make_unique<StencilDomainSplitPass>();
}
