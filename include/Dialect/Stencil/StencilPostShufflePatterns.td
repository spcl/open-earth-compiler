#ifndef MLIR_DIALECT_STENCIL_POSTSHUFFLEPATTERNS
#define MLIR_DIALECT_STENCIL_POSTSHUFFLEPATTERNS

include "mlir/Dialect/StandardOps/Ops.td"

//===----------------------------------------------------------------------===//
// Revert convertion of additions to subtractions
//===----------------------------------------------------------------------===//
def : Pat<(AddFOp (NegFOp $arg1), (NegFOp $arg2)), 
          (NegFOp (AddFOp $arg2, $arg1)),
          [],
          (addBenefit 2)>;

def : Pat<(AddFOp $arg1, (NegFOp $arg2)), 
          (SubFOp $arg1, $arg2),
          [],
          (addBenefit 1)>;

def : Pat<(AddFOp (NegFOp $arg1), $arg2), 
          (SubFOp $arg2, $arg1),
          [],
          (addBenefit 1)>;

#endif // MLIR_DIALECT_STENCIL_POSTSHUFFLEPATTERNS
