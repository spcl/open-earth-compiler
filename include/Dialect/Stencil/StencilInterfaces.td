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
    InterfaceMethod<"/*Clear the optional range attributes*/",
      "void", "clearInferredShape", (ins), [{
        if(isOptionalArrayAttr($_op.lb()) && isOptionalArrayAttr($_op.ub())) {
          $_op.removeAttr($_op.getLBAttrName());
          $_op.removeAttr($_op.getUBAttrName());
        }
    }]>,
    InterfaceMethod<"/*Set the shape of the operation*/",
      "void", "updateShape", (ins "ArrayRef<int64_t>":$lb, "ArrayRef<int64_t>":$ub), [{
        // Update the shape
        SmallVector<Attribute, kIndexSize> lbAttr, ubAttr;
        llvm::transform(lb, std::back_inserter(lbAttr), [&](int64_t x) {
          return IntegerAttr::get(IntegerType::get(64, $_op.getContext()), x);
        });
        llvm::transform(ub, std::back_inserter(ubAttr), [&](int64_t x) {
          return IntegerAttr::get(IntegerType::get(64, $_op.getContext()), x);
        });
        $_op.lbAttr(ArrayAttr::get(lbAttr, $_op.getContext()));
        $_op.ubAttr(ArrayAttr::get(ubAttr, $_op.getContext()));
        // Compute the shape
        SmallVector<int64_t, 3> shape;
        llvm::transform(llvm::zip(lb, ub), std::back_inserter(shape),
          [](std::tuple<int64_t, int64_t> x) { 
            return std::get<1>(x) - std::get<0>(x); 
          });
        // Update the result types
        for (auto result : $_op.getOperation()->getResults()) {
          auto oldType = result.getType().template cast<TempType>();
          assert(oldType.getRank() == shape.size() &&
                "expected result type to have operation rank");
          for(auto en : llvm::enumerate(oldType.getShape())) {
            if (GridType::isScalar(en.value()))
              shape[en.index()] = GridType::kScalarDimension;
          }
          auto newType = TempType::get(oldType.getElementType(), shape);
          result.setType(newType);
        }
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
    InterfaceMethod<"/*Update the argument types*/",
      "void", "updateArgumentTypes", (ins ), /*methodBody=*/[{}], [{
        // empty default implementation
    }]>,
  ];
}

def OffsetOp : OpInterface<"OffsetOp"> {
  let description = [{
    Interface to get the operation offset.
  }];

  let methods = [
    InterfaceMethod<"/*Get the offset of the operation*/",
      "Index", "getOffset", (ins), [{
        Index result;
        ArrayAttr offset = $_op.offset();
        for (auto &elem : offset)
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return result;
    }]>,
  ];
}

def ShiftOp : OpInterface<"ShiftOp"> {
  let description = [{
    Interface to shift the operation offset.
  }];

  let methods = [
    InterfaceMethod<"/*Shift operation by a constant offset*/",
      "void", "shiftByOffset", (ins "ArrayRef<int64_t>":$offset), [{}],
        /*defaultImplementation=*/[{
        SmallVector<Attribute, kIndexSize> offsetAttr;
        for (auto elem : llvm::zip(offset, $_op.offset())) {
          offsetAttr.push_back(IntegerAttr::get(
              IntegerType::get(64, $_op.getContext()),
              std::get<1>(elem).template cast<IntegerAttr>().getValue().getSExtValue() + 
              std::get<0>(elem)));
        }
        $_op.offsetAttr(ArrayAttr::get(offsetAttr, $_op.getContext()));
    }]>,
  ];
}
def ExtentOp : OpInterface<"ExtentOp"> {
  let description = [{
    Interface to get the access extent.
  }];

  let methods = [
    InterfaceMethod<"/*Get the accessed temporary*/",
      "Value", "getTemp", (ins), [{
        return $_op.temp();
    }]>,
    InterfaceMethod<"/*Get the access extent*/",
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
