#ifndef MLIR_DIALECT_STENCIL_PRESHUFFLEPATTERNS
#define MLIR_DIALECT_STENCIL_PRESHUFFLEPATTERNS

include "mlir/Dialect/StandardOps/Ops.td"

//===----------------------------------------------------------------------===//
// Convert subtractions to additions
//===----------------------------------------------------------------------===//
def : Pat<(SubFOp $arg1, $arg2), 
          (AddFOp $arg1, (NegFOp $arg2)),
          [],
          (addBenefit 1)>;

def : Pat<(NegFOp (AddFOp $arg1, $arg2)), 
          (AddFOp (NegFOp $arg1), (NegFOp $arg2)),
          [],
          (addBenefit 2)>;

def : Pat<(AddFOp (NegFOp (NegFOp $arg1)), $arg2), 
          (AddFOp $arg1, $arg2),
          [],
          (addBenefit 3)>;

def : Pat<(AddFOp $arg1, (NegFOp (NegFOp $arg2))), 
          (AddFOp $arg1, $arg2),
          [],
          (addBenefit 3)>;

#endif // MLIR_DIALECT_STENCIL_PRESHUFFLEPATTERNS
