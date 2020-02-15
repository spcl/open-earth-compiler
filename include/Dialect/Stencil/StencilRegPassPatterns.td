#ifndef MLIR_DIALECT_STENCIL_PATTERNS
#define MLIR_DIALECT_STENCIL_PATTERNS

include "mlir/Dialect/StandardOps/Ops.td"

def CanMoveArithmeticOp : CPred<[{ canMoveArithmeticOp(op) }]>;
def IsProducedBefore : CPred<[{ isProducedBefore($0, $1) }]>;

class MoveArithmeticOp<string OpType> : NativeCodeCall<
  "moveArithmeticOp<" # OpType # ">($_builder, op)">;


//===----------------------------------------------------------------------===//
// Use commutativity to move operations closer to producer
//===----------------------------------------------------------------------===//
def : Pat<(AddFOp (AddFOp $arg1, $arg2), $arg3), 
          (AddFOp (AddFOp $arg1, $arg3), $arg2),
          [
            (Constraint<IsProducedBefore> $arg1, $arg2),
            (Constraint<IsProducedBefore> $arg3, $arg2)
          ],
          (addBenefit 3)>;

def : Pat<(AddFOp (AddFOp $arg1, $arg2), $arg3), 
          (AddFOp (AddFOp $arg3, $arg2), $arg1),
          [
            (Constraint<IsProducedBefore> $arg2, $arg1),
            (Constraint<IsProducedBefore> $arg3, $arg1)
          ],
          (addBenefit 3)>;

//===----------------------------------------------------------------------===//
// Move operations closer to producer
//===----------------------------------------------------------------------===//
def : Pat<(AddFOp:$op $_, $_), 
          (MoveArithmeticOp<"AddFOp">),
          [(Constraint<CanMoveArithmeticOp>)],
          (addBenefit 1)>;

def : Pat<(SubFOp:$op $_, $_), 
          (MoveArithmeticOp<"SubFOp">),
          [(Constraint<CanMoveArithmeticOp>)],
          (addBenefit 1)>;

def : Pat<(MulFOp:$op $_, $_), 
          (MoveArithmeticOp<"MulFOp">),
          [(Constraint<CanMoveArithmeticOp>)],
          (addBenefit 2)>;

def : Pat<(DivFOp:$op $_, $_), 
          (MoveArithmeticOp<"DivFOp">),
          [(Constraint<CanMoveArithmeticOp>)],
          (addBenefit 1)>;

def : Pat<(CmpFOp:$op $_, $_), 
          (MoveArithmeticOp<"CmpFOp">),
          [(Constraint<CanMoveArithmeticOp>)],
          (addBenefit 1)>;

def : Pat<(SelectOp:$op $_, $_, $_), 
          (MoveArithmeticOp<"SelectOp">),
          [(Constraint<CanMoveArithmeticOp>)],
          (addBenefit 1)>;

// TODO this does not converge since many patterns may apply


#endif // MLIR_DIALECT_STENCIL_PATTERNS
