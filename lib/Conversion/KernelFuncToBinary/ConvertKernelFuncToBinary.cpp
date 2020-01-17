//===- ConvertKernelFuncToCubin.cpp - MLIR GPU lowering passes ------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to compile the gpu kernel functions. Currently
// only translates the function itself but no dependencies. The pass annotates
// all kernels with the resulting CUBIN string.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/NVVMIR.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;

namespace {
// Name of the attribute storing the ptx
static constexpr const char *kPtxAnnotation = "nvvm.ptx";

/// A pass converting tagged kernel modules to ptx.
class KernelToBinaryPass
    : public OperationPass<KernelToBinaryPass, gpu::GPUModuleOp> {

public:
  KernelToBinaryPass() {}

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();

    // Make sure the NVPTX target is initialized.
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    auto llvmModule = translateModuleToNVVMIR(module);
    if (!llvmModule)
      return signalPassFailure();

    // Translate the module to CUBIN and attach the result as attribute
    if (auto ptxAttr = translateGPUModuleToPtxAnnotation(
            *llvmModule, module.getLoc(), module.getName()))
      module.setAttr(kPtxAnnotation, ptxAttr);
    else
      signalPassFailure();
  }

private:
  std::string translateModuleToPtx(llvm::Module &module,
                                   llvm::TargetMachine &target_machine);

  std::string convertModuleToPtx(llvm::Module &llvmModule, Location loc,
                                 StringRef name);

  StringAttr translateGPUModuleToPtxAnnotation(llvm::Module &llvmModule,
                                               Location loc, StringRef name);
};

} // namespace

std::string
KernelToBinaryPass::translateModuleToPtx(llvm::Module &module,
                                         llvm::TargetMachine &target_machine) {
  std::string ptx;
  {
    llvm::raw_string_ostream stream(ptx);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegen_passes;
    target_machine.addPassesToEmitFile(codegen_passes, pstream, nullptr,
                                       llvm::CGFT_AssemblyFile);
    codegen_passes.run(module);
  }

  return ptx;
}

std::string KernelToBinaryPass::convertModuleToPtx(llvm::Module &llvmModule,
                                                   Location loc,
                                                   StringRef name) {
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  {
    std::string error;
    // TODO(herhut): Make triple configurable.
    constexpr const char *cudaTriple = "nvptx64-nvidia-cuda";
    llvm::Triple triple(cudaTriple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      emitError(loc, "cannot initialize target triple");
      return {};
    }
    targetMachine.reset(
        target->createTargetMachine(triple.str(), "sm_35", "+ptx60", {}, {}));
  }

  // Set the data layout of the llvm module to match what the ptx target
  llvmModule.setDataLayout(targetMachine->createDataLayout());
  return translateModuleToPtx(llvmModule, *targetMachine);
}

StringAttr KernelToBinaryPass::translateGPUModuleToPtxAnnotation(
    llvm::Module &llvmModule, Location loc, StringRef name) {
  auto ptx = convertModuleToPtx(llvmModule, loc, name);

  return StringAttr::get(ptx.c_str(), loc->getContext());
}

static PassRegistration<KernelToBinaryPass>
    pass("gpu-kernel-to-ptx", "Convert all kernel functions to PTX");
