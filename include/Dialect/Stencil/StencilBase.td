#ifndef Stencil_BASE
#define Stencil_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Stencil Dialect
//===----------------------------------------------------------------------===//

def Stencil_Dialect : Dialect {
    let name = "stencil";
    let description = [{A stencil dialect for weather and climate.}];
    let cppNamespace = "::mlir::stencil";
}

//===----------------------------------------------------------------------===//
// Stencil Types
//===----------------------------------------------------------------------===//

def Stencil_IsTempType : CPred<"$_self.isa<::mlir::stencil::TempType>()">;
def Stencil_IsFieldType : CPred<"$_self.isa<::mlir::stencil::FieldType>()">;
def Stencil_IsResultType : CPred<"$_self.isa<::mlir::stencil::ResultType>()">;

def Stencil_Temp : Type<Stencil_IsTempType, "a temporary value">;
def Stencil_Field : Type<Stencil_IsFieldType, "a input/output array">;
def Stencil_Result : Type<Stencil_IsResultType, "a result">;

def Stencil_Element : AnyTypeOf<[F32, F64]>;

def Stencil_Index : Confined<I64ArrayAttr, [ArrayCount<3>]>;
def Stencil_Loop : Confined<I64ArrayAttr, [ArrayCount<4>]>;

//===----------------------------------------------------------------------===//
// Stencil Operation
//===----------------------------------------------------------------------===//

// Base class for all Stencil ops.
class Stencil_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Stencil_Dialect, mnemonic, traits> {

    // For each Stencil op, the following static functions need to be defined in
    // StencilOps.cpp:
    //
    // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
    //                                               OperationState &state);
    // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
    // * static LogicalResult verify(<op-c++-class-name> op)
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(*this, p); }];
    let verifier = [{ return ::verify(*this); }];
}

#endif // Stencil_BASE
