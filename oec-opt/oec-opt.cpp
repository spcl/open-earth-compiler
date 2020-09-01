//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Copyright 2019 The MLIR Authors.
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
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LoopsToGPU/Passes.h"
#include "Conversion/StencilToStandard/Passes.h"
#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/MlirOptMain.h"

using namespace mlir;

// Register the parallel loop gpu mapping pass
namespace mlir {
void registerTestGpuParallelLoopMappingPass();
}

int main(int argc, char **argv) {
  registerAllDialects();
  registerAllPasses();
  registerTestGpuParallelLoopMappingPass();

  // Register the stencil passes
  registerStencilPasses();
  registerStencilConversionPasses();

  // Register the stencil pipelines
#ifdef CUDA_BACKEND_ENABLED
  registerGPUToCUBINPipeline();
#endif 
#ifdef ROCM_BACKEND_ENABLED
  registerGPUToHSACOPipeline();
#endif 

  mlir::DialectRegistry registry;
  registry.insert<stencil::StencilDialect>();
  registry.insert<StandardOpsDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<gpu::GPUDialect>();

  return failed(
      mlir::MlirOptMain(argc, argv, "Open Earth Compiler driver\n", registry));
}
