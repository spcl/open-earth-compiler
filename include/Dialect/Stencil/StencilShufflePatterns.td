#ifndef MLIR_DIALECT_STENCIL_SHUFFLEPATTERNS
#define MLIR_DIALECT_STENCIL_SHUFFLEPATTERNS

include "mlir/Dialect/StandardOps/Ops.td"

//def IsProducedBefore : Constraint<CPred<"isProducedBefore($0, $1)">>;
def IsAccessedBefore : Constraint<CPred<"isAccessedBefore($0, $1)">>;
def HasOneUse: Constraint<CPred<"$0.hasOneUse()">>;

// def IsProducedByAddFOrSubFOp : Constraint<
//   CPred<"isProducedBy<AddFOp>($0) || isProducedBy<SubFOp>($0)">>;
// def IsProducedByMulFOp : Constraint<
//   CPred<"isProducedBy<MulFOp>($0)">>;

def IsNotProducedByAddFOrSubFOp : Constraint<
  CPred<"!(isProducedBy<AddFOp>($0) || isProducedBy<SubFOp>($0))">>;
def IsNotProducedByMulFOp : Constraint<
  CPred<"!(isProducedBy<MulFOp>($0))">>;

//===----------------------------------------------------------------------===//
// Patterns linearizing the expression tree
//===----------------------------------------------------------------------===//
def : Pat<(AddFOp $arg1, (AddFOp $arg2, $arg3)), 
          (AddFOp (AddFOp $arg2, $arg3), $arg1),
          [
            (IsNotProducedByAddFOrSubFOp $arg1) 
          ],
          (addBenefit 11)>;

def : Pat<(SubFOp $arg1, (SubFOp $arg2, $arg3)), 
          (AddFOp (SubFOp $arg3, $arg2), $arg1),
          [
            (IsNotProducedByAddFOrSubFOp $arg1) 
          ],
          (addBenefit 11)>;

def : Pat<(AddFOp $arg1, (AddFOp:$res $arg2, $arg3)), 
          (AddFOp (AddFOp $arg1, $arg3), $arg2),
          [
            (IsNotProducedByAddFOrSubFOp $arg2), 
            (HasOneUse $res)
          ],
          (addBenefit 10)>;

def : Pat<(SubFOp $arg1, (SubFOp:$res $arg2, $arg3)), 
          (SubFOp (AddFOp $arg1, $arg3), $arg2),
          [
            (IsNotProducedByAddFOrSubFOp $arg2), 
            (HasOneUse $res)
          ],
          (addBenefit 10)>;

def : Pat<(AddFOp $arg1, (SubFOp:$res $arg2, $arg3)), 
          (AddFOp (SubFOp $arg1, $arg3), $arg2),
          [
            (IsNotProducedByAddFOrSubFOp $arg2), 
            (HasOneUse $res)
          ],
          (addBenefit 10)>;

def : Pat<(SubFOp $arg1, (AddFOp:$res $arg2, $arg3)), 
          (SubFOp (SubFOp $arg1, $arg3), $arg2),
          [
            (IsNotProducedByAddFOrSubFOp $arg2), 
            (HasOneUse $res)
          ],
          (addBenefit 10)>;

def : Pat<(MulFOp $arg1, (MulFOp $arg2, $arg3)), 
          (MulFOp (MulFOp $arg2, $arg3), $arg1),
          [
            (IsNotProducedByMulFOp $arg1) 
          ],
          (addBenefit 11)>;

def : Pat<(MulFOp $arg1, (MulFOp:$res $arg2, $arg3)), 
          (MulFOp (MulFOp $arg1, $arg3), $arg2),
          [
            (IsNotProducedByAddFOrSubFOp $arg2), 
            (HasOneUse $res)
          ],
          (addBenefit 10)>;

//===----------------------------------------------------------------------===//
// Patterns exploiting commutativity
//===----------------------------------------------------------------------===//
def : Pat<(AddFOp (AddFOp:$res $arg3, $arg2), $arg1),
          (AddFOp (AddFOp $arg3, $arg1), $arg2),
          [
            (IsAccessedBefore $arg1, $arg2),
            (HasOneUse $res)
          ],
          (addBenefit 0)>;

def : Pat<(SubFOp (SubFOp:$res $arg3, $arg2), $arg1),
          (SubFOp (SubFOp $arg3, $arg1), $arg2),
          [
            (IsAccessedBefore $arg1, $arg2),
            (HasOneUse $res)
          ],
          (addBenefit 0)>;

def : Pat<(AddFOp (SubFOp:$res $arg3, $arg2), $arg1),
          (SubFOp (AddFOp $arg3, $arg1), $arg2),
          [
            (IsAccessedBefore $arg1, $arg2),
            (HasOneUse $res)
          ],
          (addBenefit 0)>;

def : Pat<(SubFOp (AddFOp:$res $arg3, $arg2), $arg1),
          (AddFOp (SubFOp $arg3, $arg1), $arg2),
          [
            (IsAccessedBefore $arg1, $arg2),
            (HasOneUse $res)
          ],
          (addBenefit 0)>;

def : Pat<(MulFOp (MulFOp:$res $arg3, $arg2), $arg1),
          (MulFOp (MulFOp $arg3, $arg1), $arg2),
          [
            (IsAccessedBefore $arg1, $arg2),
            (HasOneUse $res)
          ],
          (addBenefit 0)>;

#endif // MLIR_DIALECT_STENCIL_SHUFFLEPATTERNS
