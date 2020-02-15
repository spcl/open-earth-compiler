#ifndef MLIR_DIALECT_STENCIL_PATTERNS
#define MLIR_DIALECT_STENCIL_PATTERNS

include "mlir/Dialect/StandardOps/Ops.td"

def CanMoveArithmeticOp : CPred<[{ canMoveArithmeticOp(op) }]>;

class MoveArithmeticOp<string OpType> : NativeCodeCall<
  "moveArithmeticOp<" # OpType # ">($_builder, op)">;

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
          (addBenefit 1)>;

#endif // MLIR_DIALECT_STENCIL_PATTERNS
