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
        Optional<ArrayAttr> lb = $_op.lb();
        for (auto &elem : lb.getValue()) 
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return result;
    }]>,
    InterfaceMethod<"/*Get the upper bound of the operation*/",
      "Index", "getUB", (ins), [{
        Index result;
        Optional<ArrayAttr> ub = $_op.ub();
        for (auto &elem : ub.getValue()) 
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return result;
    }]>,
      InterfaceMethod<"/*Set the lower bound of the operation*/",
      "void", "setLB", (ins "ArrayRef<int64_t>":$lb), [{
        SmallVector<Attribute, kIndexSize> lbAttr;
        for (size_t i = 0, e = lb.size(); i != e; ++i) { 
          lbAttr.push_back(IntegerAttr::get(IntegerType::get(64, $_op.getContext()), lb[i]));
        }
        $_op.lbAttr(ArrayAttr::get(lbAttr, $_op.getContext()));
    }]>,
    InterfaceMethod<"/*Set the upper bound of the operation*/",
      "void", "setUB", (ins "ArrayRef<int64_t>":$ub), [{
        SmallVector<Attribute, kIndexSize> ubAttr;
        for (size_t i = 0, e = ub.size(); i != e; ++i) { 
          ubAttr.push_back(IntegerAttr::get(IntegerType::get(64, $_op.getContext()), ub[i]));
        }
        $_op.ubAttr(ArrayAttr::get(ubAttr, $_op.getContext()));
    }]>,
    InterfaceMethod<"/*Verify if the operation has valid bounds*/",
       "bool", "hasShape", (ins), [{
        Optional<ArrayAttr> lb = $_op.lb();
        Optional<ArrayAttr> ub = $_op.ub();
        return lb.hasValue() && ub.hasValue();
    }]>,   
    InterfaceMethod<"/*Get the rank of the operation*/",
      "int64_t", "getRank", (ins), [{
        Optional<ArrayAttr> lb = $_op.lb();
        Optional<ArrayAttr> ub = $_op.ub();
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
        ArrayAttr offset = $_op.offset();
        for (auto &elem : offset.getValue())
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return result;
    }]>,
    InterfaceMethod<"/*Set the offset of the operation*/",
      "void", "setOffset", (ins "ArrayRef<int64_t>":$index), [{
        SmallVector<Attribute, kIndexSize> offsetAttr;
        for(auto value : index)
          offsetAttr.push_back(IntegerAttr::get(IntegerType::get(64, $_op.getContext()), value));
        $_op.offsetAttr(ArrayAttr::get(offsetAttr, $_op.getContext()));
    }]>,
  ];
}

def ExtentOp : OpInterface<"ExtentOp"> {
  let description = [{
    Interface to get the access extent.
  }];

  let methods = [
    InterfaceMethod<"/*Get the offset of the operation*/",
      "Value", "getAccessedValue", (ins), [{}],
        /*defaultImplementation=*/[{
        return $_op.temp();
    }]>,
    InterfaceMethod<"/*Get the offset of the operation*/",
      "std::tuple<Index, Index>", "getAccessExtent", (ins), [{}],
        /*defaultImplementation=*/[{
        Index result;
        ArrayAttr offset = $_op.offset();
        for (auto &elem : offset.getValue())
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return std::make_tuple(result, result);
    }]>,
  ];
}

#endif // Stencil_INTERFACES
