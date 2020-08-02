//===- RandomStencilGenerator.h ---------------------------------*- C++ -*-===//
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
// ============================================================================
//
// This file implements random stencil program generation engine.
//
//===---------------------------------------------------------------------===//

#ifndef MLIR_TARGET_STENCIL_RANDOMSTENCILGENERATOR_H
#define MLIR_TARGET_STENCIL_RANDOMSTENCILGENERATOR_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

#include "llvm/Support/raw_ostream.h"

#include "MarkovChain.h"
#include "RandGenerator.h"
#include "BucketDistribution.h"
//#include "mlir/Target/Stencil/DCE.h"

#include <map>
#include <set>
#include <stack>
#include <string>
#include <sstream>
#include <algorithm>

#endif MLIR_TARGET_STENCIL_RANDOMSTENCILGENERATOR_H