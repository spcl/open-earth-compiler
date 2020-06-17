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

def OffsetOp : OpInterface<"OffsetOp"> {
  let description = [{
    Interface to get and set offsets.
  }];

  let methods = [
    InterfaceMethod<"/*Get the offset of the operation*/",
      "Index", "getOffset", (ins), [{
        Index result;
        ArrayAttr offset = op.offset();
        for (auto &elem : offset.getValue())
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return result;
    }]>,
    InterfaceMethod<"/*Set the offset of the operation*/",
      "void", "setOffset", (ins "ArrayRef<int64_t>":$index), [{
        SmallVector<Attribute, kIndexSize> offsetAttr;
        for(auto value : index)
          offsetAttr.push_back(IntegerAttr::get(IntegerType::get(64, op.getContext()), value));
        op.offsetAttr(ArrayAttr::get(offsetAttr, op.getContext()));
    }]>,
    InterfaceMethod<"/*Set the operand type*/",
      "void", "setOperandShape", (ins  "Value":$operand, "TempType":$newType), /*methodBody=*/[{}], [{
        // empty default implementation
    }]>,
  ];
}

#endif // Stencil_INTERFACES
