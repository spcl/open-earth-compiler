#ifndef MLIR_DIALECT_STENCIL_SHUFFLEPATTERNS
#define MLIR_DIALECT_STENCIL_SHUFFLEPATTERNS

include "mlir/Dialect/StandardOps/Ops.td"

def IsProducedBefore : CPred<[{ isProducedBefore($0, $1) }]>;

def HasSingleUse : CPred<[{ hasSingleUse($0) }]>;

//===----------------------------------------------------------------------===//
// Patterns to linearize expression tree
//===----------------------------------------------------------------------===//
def : Pat<(AddFOp (AddFOp:$res $arg1, $arg2), (AddFOp $arg3, $arg4)), 
          (AddFOp (AddFOp (AddFOp $arg4, $arg3), $arg2), $arg1),
          [(Constraint<HasSingleUse> $res)],
          (addBenefit 10)>;

//===----------------------------------------------------------------------===//
// Patterns to sort arguments by production order
//===----------------------------------------------------------------------===//
// def : Pat<(AddFOp $arg1, $arg2), 
//           (AddFOp $arg2, $arg1),
//           [(Constraint<IsProducedBefore> $arg1, $arg2)],
//           (addBenefit 10)>;

//===----------------------------------------------------------------------===//
// Patterns to sort arguments by production order
//===----------------------------------------------------------------------===//

def : Pat<(AddFOp (AddFOp:$res $arg3, $arg2), $arg1),
          (AddFOp (AddFOp $arg3, $arg1), $arg2),
          [
            (Constraint<IsProducedBefore> $arg1, $arg2),
            (Constraint<HasSingleUse> $res)
          ],
          (addBenefit 0)>;

// def : Pat<(AddFOp $arg1, (AddFOp:$res $arg2, $arg3)),
//           (AddFOp $arg3, (AddFOp $arg2, $arg1)),
//           [
//             (Constraint<IsProducedBefore> $arg1, $arg3),
//             (Constraint<IsProducedBefore> $arg2, $arg3),
//             (Constraint<HasSingleUse> $res)
//           ],
//           (addBenefit 0)>;


// TODO only rewrite if temporary has single use

// Pattern moves shortest dependency outward
// def : Pat<(AddFOp $arg1, (AddFOp:$res $arg2, $arg3)),
//           (AddFOp $arg2, (AddFOp $arg1, $arg3)),
//           [
//             (Constraint<IsProducedBefore> $arg1, $arg2),
//             (Constraint<IsProducedBefore> $arg3, $arg2),
//             (Constraint<HasSingleUse> $res)
//           ],
//           (addBenefit 1)>;

// def : Pat<(AddFOp $arg1, (AddFOp:$res $arg2, $arg3)),
//           (AddFOp $arg3, (AddFOp $arg2, $arg1)),
//           [
//             (Constraint<IsProducedBefore> $arg1, $arg3),
//             (Constraint<IsProducedBefore> $arg2, $arg3),
//             (Constraint<HasSingleUse> $res)
//           ],
//           (addBenefit 1)>;

// def : Pat<(AddFOp (AddFOp:$res $arg2, $arg3), $arg1),
//           (AddFOp (AddFOp $arg1, $arg3), $arg2),
//           [
//             (Constraint<IsProducedBefore> $arg1, $arg2),
//             (Constraint<IsProducedBefore> $arg3, $arg2),
//             (Constraint<HasSingleUse> $res)
//           ],
//           (addBenefit 1)>;

// def : Pat<(AddFOp (AddFOp:$res $arg2, $arg3), $arg1),
//           (AddFOp (AddFOp $arg2, $arg1), $arg3),
//           [
//             (Constraint<IsProducedBefore> $arg1, $arg3),
//             (Constraint<IsProducedBefore> $arg2, $arg3),
//             (Constraint<HasSingleUse> $res)
//           ],
//           (addBenefit 1)>;

// def : Pat<(AddFOp $arg1, (AddFOp $arg2, $arg3)),
//           (AddFOp $arg3, (AddFOp $arg2, $arg1)),
//           [
//             (Constraint<IsProducedLast> $arg3, $arg1, $arg2)
//           ],
//           (addBenefit 1)>;

// def : Pat<(AddFOp $arg1, (AddFOp $arg2, $arg3)),
//           (AddFOp $arg3, (AddFOp $arg1, $arg3)),
//           [
//             (Constraint<isProducedBefore> $arg1, $arg3)
//           ],
//           (addBenefit 1)>;

// def : Pat<(AddFOp (AddFOp $arg2, $arg3), $arg1),
//           (AddFOp (AddFOp $arg1, $arg3), $arg2),
//           [
//             (Constraint<isProducedBefore> $arg1, $arg2)
//           ],
//           (addBenefit 1)>;

// def : Pat<(AddFOp (AddFOp $arg2, $arg3), $arg1),
//           (AddFOp (AddFOp $arg1, $arg3), $arg3),
//           [
//             (Constraint<isProducedBefore> $arg1, $arg3)
//           ],
//           (addBenefit 1)>;


// def : Pat<(MulFOp $arg1, $arg2), 
//           (MulFOp $arg2, $arg1),
//           [(Constraint<isProducedBefore> $arg2, $arg1)],
//           (addBenefit 2)>;

// def : Pat<(MulFOp $arg1, (MulFOp $arg2, $arg3)),
//           (MulFOp $arg3, (MulFOp $arg2, $arg1)),
//           [(Constraint<isProducedBefore> $arg1, $arg3)],
//           (addBenefit 1)>;

// add patterns for multiplication and division?
// (keep subtraction or not)
// iterate between commutativity patterns and scheduling?


//===----------------------------------------------------------------------===//
// Move operations closer to producer
//===----------------------------------------------------------------------===//

// def CanMoveArithmeticOp : CPred<[{ canMoveArithmeticOp(op) }]>;

// class MoveArithmeticOp<string OpType> : NativeCodeCall<
//   "moveArithmeticOp<" # OpType # ">($_builder, op)">;

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
