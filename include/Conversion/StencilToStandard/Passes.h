#ifndef CONVERSION_STENCILTOSTANDARD_PASSES_H
#define CONVERSION_STENCILTOSTANDARD_PASSES_H

#include <memory>

namespace mlir {

class Pass;

std::unique_ptr<Pass> createConvertStencilToStandardPass();

} // namespace mlir

#endif // CONVERSION_STENCILTOSTANDARD_PASSES_H
