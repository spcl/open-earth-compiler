#ifndef Stencil_BASE
#define Stencil_BASE

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// Stencil dialect definition
//===----------------------------------------------------------------------===//

def Stencil_Dialect : Dialect {
    let name = "stencil";

    let description = [{
        A simple stencil dialect in MLIR.
    }];

    let cppNamespace = "stencil";
}

//===----------------------------------------------------------------------===//
// Stencil type definitions
//===----------------------------------------------------------------------===//

def Stencil_IsViewType : CPred<"$_self.isa<::mlir::stencil::ViewType>()">;
def Stencil_IsFieldType : CPred<"$_self.isa<::mlir::stencil::FieldType>()">;

def Stencil_View : Type<Stencil_IsViewType, "a view on a field">;
def Stencil_Field : Type<Stencil_IsFieldType, "a field">;
def Stencil_ElementType : AnyTypeOf<[I64, F64]>;

//===----------------------------------------------------------------------===//
// Stencil op definition
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
