#ifndef Stencil_INTERFACES
#define Stencil_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Stencil Interfaces
//===----------------------------------------------------------------------===//

def ShapeOp : OpInterface<"ShapeOp"> {
  let description = [{
    Interface to get the operation bounds.
  }];

  let methods = [
    InterfaceMethod<"/*Get the lower bound of the operation*/",
      "Index", "getLB", (ins), [{
        Index result;
        Optional<ArrayAttr> lb = op.lb();
        for (auto &elem : lb.getValue()) 
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return result;
    }]>,
    InterfaceMethod<"/*Get the upper bound of the operation*/",
      "Index", "getUB", (ins), [{
        Index result;
        Optional<ArrayAttr> ub = op.ub();
        for (auto &elem : ub.getValue()) 
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return result;
    }]>,
      InterfaceMethod<"/*Set the lower bound of the operation*/",
      "void", "setLB", (ins "ArrayRef<int64_t>":$lb), [{
        SmallVector<Attribute, kIndexSize> lbAttr;
        for (size_t i = 0, e = lb.size(); i != e; ++i) { 
          lbAttr.push_back(IntegerAttr::get(IntegerType::get(64, op.getContext()), lb[i]));
        }
        op.lbAttr(ArrayAttr::get(lbAttr, op.getContext()));
    }]>,
    InterfaceMethod<"/*Set the upper bound of the operation*/",
      "void", "setUB", (ins "ArrayRef<int64_t>":$ub), [{
        SmallVector<Attribute, kIndexSize> ubAttr;
        for (size_t i = 0, e = ub.size(); i != e; ++i) { 
          ubAttr.push_back(IntegerAttr::get(IntegerType::get(64, op.getContext()), ub[i]));
        }
        op.ubAttr(ArrayAttr::get(ubAttr, op.getContext()));
    }]>,
    InterfaceMethod<"/*Verify if the operation has valid bounds*/",
       "bool", "hasShape", (ins), [{
        Optional<ArrayAttr> lb = op.lb();
        Optional<ArrayAttr> ub = op.ub();
        return lb.hasValue() && ub.hasValue();
    }]>,   
    InterfaceMethod<"/*Get the rank of the operation*/",
      "int64_t", "getRank", (ins), [{
        Optional<ArrayAttr> lb = op.lb();
        Optional<ArrayAttr> ub = op.ub();
        assert(lb.getValue().size() == ub.getValue().size() && 
               "expected lower and upper bound to have the same rank");
        return lb.getValue().size();
    }]>,
    InterfaceMethod<"/*Set the operand type*/",
      "void", "setOperandShape", (ins  "Value":$operand, "TempType":$newType), /*methodBody=*/[{}], [{
        // empty default implementation
    }]>,
  ];
}

// def ShapeInference : OpInterface<"ShapeInference"> {
//   let description = [{
//     Interface to set the operation bounds and result types.
//   }];

//   let methods = [
//     InterfaceMethod<"/*Method to get the bounds of the operation.*/",
//       "void", "setOpShape", (ins "ArrayRef<int64_t>":$lb, "ArrayRef<int64_t>":$ub), [{
//         assert(lb.size() == ub.size() && 
//                "expected bounds to have the same size");
//         SmallVector<Attribute, kIndexSize> lbAttr;
//         SmallVector<Attribute, kIndexSize> ubAttr;
//         for (size_t i = 0, e = lb.size(); i != e; ++i) { 
//           lbAttr.push_back(IntegerAttr::get(IntegerType::get(64, op.getContext()), lb[i]));
//           ubAttr.push_back(IntegerAttr::get(IntegerType::get(64, op.getContext()), ub[i]));
//         }
//         op.lbAttr(ArrayAttr::get(lbAttr, op.getContext()));
//         op.ubAttr(ArrayAttr::get(ubAttr, op.getContext()));
//     }]>
//   ];
// }

#endif // Stencil_INTERFACES
