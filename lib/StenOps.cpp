// Dialect/Sten/StenOps.cpp

#include "sten/StenOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "sten/StenDialect.h"
#include "sten/StenTypes.h"

using namespace mlir;

// Custom operations go here

namespace mlir {
namespace sten {
#define GET_OP_CLASSES
#include "sten/StenOps.cpp.inc"
} // namespace sten
} // namespace mlir
