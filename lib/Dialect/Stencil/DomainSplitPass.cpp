#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace stencil;

void splitOnDomains(FuncOp funcOp) {
  // Maps to remember domains associated with every operation
  llvm::DenseMap<Operation *, Index> negative;
  llvm::DenseMap<Operation *, Index> positive;

  // Go through the operations in reverse order
  Block &entryBlock = funcOp.getOperation()->getRegion(0).front();
  for (auto op = entryBlock.rbegin(); op != entryBlock.rend(); ++op) {
    // For stores enter their domain into global domain map
    if (auto storeOp = dyn_cast<stencil::StoreOp>(*op)) {
      negative[storeOp.getOperation()] =
          cast<ShapeOp>(storeOp.getOperation()).getLB();
      positive[storeOp.getOperation()] =
          cast<ShapeOp>(storeOp.getOperation()).getUB();
      // For other ShapeOps split them if they participate in multiple domains
    }
    if (dyn_cast<ApplyOp>(*op) || dyn_cast<CombineOp>(*op) ||
        dyn_cast<BufferOp>(*op) || dyn_cast<stencil::LoadOp>(*op)) {
      // List of all domains an op participates with
      SmallVector<std::tuple<Index, Index>, 10> domains;
      // Map of what use of what result of the op participates in what domain
      DenseMap<OpOperand *, std::tuple<Index, Index>> useToDomain;
      // Loop over all results and all uses for every result
      for (auto result : op->getResults()) {
        for (auto &use : result.getUses()) {
          std::tuple<Index, Index> useDomain = {negative[use.getOwner()],
                                                positive[use.getOwner()]};
          if (auto combineOp = dyn_cast<CombineOp>(use.getOwner())) {
            if (combineOp.isLowerOperand(use.getOperandNumber())) {
              std::get<1>(useDomain)[combineOp.dim()] = combineOp.index();
            }
            if (combineOp.isUpperOperand(use.getOperandNumber())) {
              std::get<0>(useDomain)[combineOp.dim()] = combineOp.index();
            }
          }
          // Check if domain of this use of this result already exists
          if (llvm::all_of(domains, [&](std::tuple<Index, Index> tuple) {
                return std::get<0>(tuple) != std::get<0>(useDomain) ||
                       std::get<1>(tuple) != std::get<1>(useDomain);
              })) {
            // If the domain does not already exist, add the new domain
            domains.push_back(std::tuple<Index, Index>(std::get<0>(useDomain),
                                                       std::get<1>(useDomain)));
            useToDomain[&use] = std::tuple<Index, Index>(
                std::get<0>(useDomain), std::get<1>(useDomain));
          } else {
            // If the domain already exists add the use to the domain
            for (auto domain : domains) {
              if (std::get<0>(domain) == std::get<0>(useDomain) &&
                  std::get<1>(domain) == std::get<1>(useDomain)) {
                useToDomain[&use] = std::tuple<Index, Index>(
                    std::get<0>(useDomain), std::get<1>(useDomain));
              }
            }
          }
        }
      }

      OpBuilder builder(&(*op));
      builder.setInsertionPointAfter(&(*op));

      // Set domain for op in global map
      negative[&(*op)] = std::get<0>(domains[0]);
      positive[&(*op)] = std::get<1>(domains[0]);

      // For every domain construct one copy of the op
      for (size_t i = 1, e = domains.size(); i < e; i++) {
        Operation *clonedOp = builder.clone(*op);
        // Add domain of new op to global map
        negative[clonedOp] = std::get<0>(domains[i]);
        positive[clonedOp] = std::get<1>(domains[i]);
        // Replace uses of results with the op of the according domain
        for (auto res : llvm::enumerate(
                 llvm::zip(op->getResults(), clonedOp->getResults()))) {
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

void splitOnLastCombines(FuncOp funcOp) {
  // Map to remember the closest downstream stencil combine for every operation
  llvm::DenseMap<Operation *, Operation *> lastCombine;

  // Go through the operations in reverse order
  Block &entryBlock = funcOp.getOperation()->getRegion(0).front();
  for (auto op = entryBlock.rbegin(); op != entryBlock.rend(); ++op) {
    // For stores, set their closest downstream combine to nullptr
    if (auto storeOp = dyn_cast<stencil::StoreOp>(*op)) {
      lastCombine[storeOp.getOperation()] = nullptr;
      // For other operation types, split them if they are connected to more
      // than one downstream combine
    }
    if (dyn_cast<ApplyOp>(*op) || dyn_cast<CombineOp>(*op) ||
        dyn_cast<BufferOp>(*op) || dyn_cast<stencil::LoadOp>(*op)) {
      // List of all of the downstream combines an operation is connected with
      SmallVector<Operation *, 10> combines;
      // Map of what use of what result of the op is connected to which combine
      DenseMap<OpOperand *, Operation *> useToCombine;
      // Loop over all results and all uses for every result
      for (auto result : op->getResults()) {
        for (auto &use : result.getUses()) {
          // Check if combine of this use of this result already exists
          if (llvm::all_of(combines, [&](Operation *combineOp) {
                return combineOp != lastCombine[use.getOwner()];
              })) {
            // If the combine does not already exist, add it as a new combine
            combines.push_back(lastCombine[use.getOwner()]);
            useToCombine[&use] = lastCombine[use.getOwner()];
          } else {
            // If the combine already exists add the use to the list
            for (auto combine : combines) {
              if (combine == lastCombine[use.getOwner()]) {
                useToCombine[&use] = lastCombine[use.getOwner()];
              }
            }
          }
        }
      }

      OpBuilder builder(&(*op));
      builder.setInsertionPointAfter(&(*op));

      // Set combine for op in global map
      if (dyn_cast<CombineOp>(*op)) {
        lastCombine[&(*op)] = &(*op);
      } else {
        // if op itself is a combine, set itself as closest combine
        lastCombine[&(*op)] = combines[0];
      }

      // For every combine construct one copy of the op
      for (size_t i = 1, e = combines.size(); i < e; i++) {
        Operation *clonedOp = builder.clone(*op);
        // Add combine of new op to global map
        if (dyn_cast<CombineOp>(*op)) {
          lastCombine[clonedOp] = clonedOp;
        } else {
          lastCombine[clonedOp] = combines[i];
        }
        // Replace uses of results with the op of the according domain
        for (auto res : llvm::enumerate(
                 llvm::zip(op->getResults(), clonedOp->getResults()))) {
          std::get<0>(res.value())
              .replaceUsesWithIf(std::get<1>(res.value()),
                                 [&](OpOperand &operand) {
                                   return useToCombine[&operand] == combines[i];
                                 });
        }
      }
    }
  }
}

namespace {

struct DomainSplitPass : public DomainSplitPassBase<DomainSplitPass> {

  void runOnFunction() override;
};

void DomainSplitPass::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Only run on functions marked as stencil programs
  if (!StencilDialect::isStencilProgram(funcOp))
    return;

  // Check shape inference has been executed
  bool hasShapeOpWithoutShape = false;
  funcOp.walk([&](ShapeOp shapeOp) {
    if (!shapeOp.hasShape())
      hasShapeOpWithoutShape = true;
  });
  if (!hasShapeOpWithoutShape) {
    funcOp.emitOpError("execute combine split before shape inference");
    signalPassFailure();
    return;
  }

  splitOnDomains(funcOp);
  funcOp.dump();
  splitOnLastCombines(funcOp);
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createDomainSplitPass() {
  return std::make_unique<DomainSplitPass>();
}
