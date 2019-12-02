// Dialect/Sten/StenBase.td

#ifndef STEN_BASE
#define STEN_BASE

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// Sten dialect definition
//===----------------------------------------------------------------------===//

def Sten_Dialect : Dialect {
    let name = "sten";

    let description = [{
        A simple stencil dialect in MLIR.
    }];

    let cppNamespace = "sten";
}

//===----------------------------------------------------------------------===//
// Sten op definition
//===----------------------------------------------------------------------===//

// Base class for all Sten ops.
class Sten_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Sten_Dialect, mnemonic, traits> {

    // For each Sten op, the following static functions need to be defined in
    // StenOps.cpp:
    //
    // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
    //                                               OperationState &state);
    // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
    // * static LogicalResult verify(<op-c++-class-name> op)
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(*this, p); }];
    let verifier = [{ return ::verify(*this); }];
}

#endif // STEN_BASE
