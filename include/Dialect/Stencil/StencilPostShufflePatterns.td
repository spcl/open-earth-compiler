#ifndef MLIR_DIALECT_STENCIL_POSTSHUFFLEPATTERNS
#define MLIR_DIALECT_STENCIL_POSTSHUFFLEPATTERNS

include "mlir/Dialect/StandardOps/Ops.td"

def HasOneUse: Constraint<CPred<"$0.hasOneUse()">>;

def IsProducedByArithOp : Constraint<
  CPred<"(isProducedBy<AddFOp>($0) || isProducedBy<SubFOp>($0) || isProducedBy<MulFOp>($0))">>;

def IsNotProducedByArithOp : Constraint<
  CPred<"!(isProducedBy<AddFOp>($0) || isProducedBy<SubFOp>($0) || isProducedBy<MulFOp>($0))">>;

def AreNotAllProducedByArithOp : Constraint<
  CPred<"(!(isProducedBy<AddFOp>($0) || isProducedBy<SubFOp>($0) || isProducedBy<MulFOp>($0)) ||" #
        "!(isProducedBy<AddFOp>($1) || isProducedBy<SubFOp>($1) || isProducedBy<MulFOp>($1)))">>;


//===----------------------------------------------------------------------===//
// Delinearize the computation
//===----------------------------------------------------------------------===//
def : Pat<(AddFOp (AddFOp:$res $arg1, $arg2), $arg3), 
          (AddFOp $arg1, (AddFOp $arg2, $arg3)),
          [
            (IsProducedByArithOp $arg1),
            (AreNotAllProducedByArithOp $arg2, $arg3),
            (HasOneUse $res)
          ],
          (addBenefit 10)>;

def : Pat<(AddFOp (SubFOp:$res $arg1, $arg2), $arg3), 
          (AddFOp $arg1, (SubFOp $arg3, $arg2)),
          [
            (IsProducedByArithOp $arg1),
            (AreNotAllProducedByArithOp $arg2, $arg3),
            (HasOneUse $res)
          ],
          (addBenefit 10)>;

def : Pat<(SubFOp (AddFOp:$res $arg1, $arg2), $arg3), 
          (AddFOp $arg1, (SubFOp $arg2, $arg3)),
          [
            (IsProducedByArithOp $arg1),
            (AreNotAllProducedByArithOp $arg2, $arg3),
            (HasOneUse $res)
          ],
          (addBenefit 10)>;

def : Pat<(SubFOp (SubFOp:$res $arg1, $arg2), $arg3), 
          (SubFOp $arg1, (AddFOp $arg2, $arg3)),
          [
            (IsProducedByArithOp $arg1),
            (AreNotAllProducedByArithOp $arg2, $arg3),
            (HasOneUse $res)
          ],
          (addBenefit 10)>;

def : Pat<(MulFOp (MulFOp:$res $arg1, $arg2), $arg3), 
          (MulFOp $arg1, (MulFOp $arg2, $arg3)),
          [
            (IsProducedByArithOp $arg1),
            (AreNotAllProducedByArithOp $arg2, $arg3),
            (HasOneUse $res)
          ],
          (addBenefit 10)>;

#endif // MLIR_DIALECT_STENCIL_POSTSHUFFLEPATTERNS
