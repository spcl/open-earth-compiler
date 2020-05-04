#ifndef Stencil_INTERFACES
#define Stencil_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Stencil Interfaces
//===----------------------------------------------------------------------===//

// TODO
// def ShapeInterface : OpInterface<"ShapeInterface"> {
//   let description = [{
//     Interface to access the lower and upper bounds of an operation.
//   }];

//   let methods = [
//     InterfaceMethod<"/*Method returns the FieldType of the field operand or result*/",
//       "FieldType", "getFieldType", (ins), [{
//         return op.field().getType().template cast<FieldType>();
//     }]>
//   ];
// }

#endif // Stencil_INTERFACES
