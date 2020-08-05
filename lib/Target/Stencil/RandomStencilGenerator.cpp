//===- RandomStencilGenerator.cpp ------------------------------*- C++ --*-===//
//
// Copyright 2020 Jakub Lichman
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements random stencil program generation engine.
//
//===----------------------------------------------------------------------===//

#include "Target/Stencil/RandomStencilGenerator.h"

using namespace mlir;
using namespace experimental;

namespace mlir {
namespace experimental {

} // namespace experimental
} // namespace mlir

namespace mlir {
void registerRandomStencilGenerator() {
  TranslateFromMLIRRegistration registration(
    "generate-random-stencil-program",
    [](ModuleOp module, llvm::raw_ostream &output) {
      auto chain = MarkovChain(module);
      chain.print(output);

      auto pg = RandomStencilGenerator(module);
      auto p = pg.generateRandomProgramOnTheFly(output);
      output << p << "\n";
      return success();
    });
}
}