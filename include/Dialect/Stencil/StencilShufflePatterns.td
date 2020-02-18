#ifndef MLIR_DIALECT_STENCIL_SHUFFLEPATTERNS
#define MLIR_DIALECT_STENCIL_SHUFFLEPATTERNS

include "mlir/Dialect/StandardOps/Ops.td"

def IsProducedBefore : CPred<[{ isProducedBefore($0, $1) }]>;

def CanMoveArithmeticOp : CPred<[{ canMoveArithmeticOp(op) }]>;

class MoveArithmeticOp<string OpType> : NativeCodeCall<
  "moveArithmeticOp<" # OpType # ">($_builder, op)">;

//===----------------------------------------------------------------------===//
// Use commutativity to move operations closer to producer
//===----------------------------------------------------------------------===//
def : Pat<(AddFOp $arg1, $arg2), 
          (AddFOp $arg2, $arg1),
          [(Constraint<IsProducedBefore> $arg2, $arg1)],
          (addBenefit 2)>;

def : Pat<(AddFOp $arg1, (AddFOp $arg2, $arg3)),
          (AddFOp $arg3, (AddFOp $arg2, $arg1)),
          [(Constraint<IsProducedBefore> $arg1, $arg3)],
          (addBenefit 1)>;

def : Pat<(MulFOp $arg1, $arg2), 
          (MulFOp $arg2, $arg1),
          [(Constraint<IsProducedBefore> $arg2, $arg1)],
          (addBenefit 2)>;

def : Pat<(MulFOp $arg1, (MulFOp $arg2, $arg3)),
          (MulFOp $arg3, (MulFOp $arg2, $arg1)),
          [(Constraint<IsProducedBefore> $arg1, $arg3)],
          (addBenefit 1)>;

// add patterns for multiplication and division?
// (keep subtraction or not)
// iterate between commutativity patterns and scheduling?


//===----------------------------------------------------------------------===//
// Move operations closer to producer
//===----------------------------------------------------------------------===//
// def : Pat<(NegFOp:$op $_), 
//           (MoveArithmeticOp<"NegFOp">),
//           [(Constraint<CanMoveArithmeticOp>)],
//           (addBenefit 10)>;

// def : Pat<(AddFOp:$op $_, $_), 
//           (MoveArithmeticOp<"AddFOp">),
//           [(Constraint<CanMoveArithmeticOp>)],
//           (addBenefit 0)>;

// def : Pat<(MulFOp:$op $_, $_), 
//           (MoveArithmeticOp<"MulFOp">),
//           [(Constraint<CanMoveArithmeticOp>)],
//           (addBenefit 0)>;

// def : Pat<(SubFOp:$op $_, $_), 
//           (MoveArithmeticOp<"SubFOp">),
//           [(Constraint<CanMoveArithmeticOp>)],
//           (addBenefit 1)>;

// def : Pat<(MulFOp:$op $_, $_), 
//           (MoveArithmeticOp<"MulFOp">),
//           [(Constraint<CanMoveArithmeticOp>)],
//           (addBenefit 2)>;

// def : Pat<(DivFOp:$op $_, $_), 
//           (MoveArithmeticOp<"DivFOp">),
//           [(Constraint<CanMoveArithmeticOp>)],
//           (addBenefit 1)>;

// def : Pat<(CmpFOp:$op $_, $_), 
//           (MoveArithmeticOp<"CmpFOp">),
//           [(Constraint<CanMoveArithmeticOp>)],
//           (addBenefit 1)>;

// def : Pat<(SelectOp:$op $_, $_, $_), 
//           (MoveArithmeticOp<"SelectOp">),
//           [(Constraint<CanMoveArithmeticOp>)],
//           (addBenefit 1)>;

// TODO this does not converge since many patterns may apply


#endif // MLIR_DIALECT_STENCIL_SHUFFLEPATTERNS
