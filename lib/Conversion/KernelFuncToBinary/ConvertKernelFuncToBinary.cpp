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
#include "llvm/Target/TargetMachine.h"

#include "cuda.h"

using namespace mlir;

namespace {
// Name of the attribute storing the cubin
static constexpr const char *kCubinAnnotation = "nvvm.cubin";

/// A pass converting tagged kernel modules to cubin blobs.
///
/// If tagged as a kernel module, each contained function is translated to NVVM
/// IR and further to PTX.
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

    // Translate the module to CUBIN and attach the result as attribute to the
    // module.
    if (auto cubinAttr = translateGPUModuleToCubinAnnotation(
            *llvmModule, module.getLoc(), module.getName()))
      module.setAttr(kCubinAnnotation, cubinAttr);
    else
      signalPassFailure();
  }

private:
  static OwnedCubin compilePtxToCubin(const std::string &ptx, Location,
                                      StringRef);

  std::string translateModuleToPtx(llvm::Module &module,
                                   llvm::TargetMachine &target_machine);

  /// Converts llvmModule to cubin using the user-provided generator. Location
  /// is used for error reporting and name is forwarded to the CUBIN generator
  /// to use in its logging mechanisms.
  OwnedCubin convertModuleToCubin(llvm::Module &llvmModule, Location loc,
                                  StringRef name);

  /// Translates llvmModule to cubin and returns the result as attribute.
  StringAttr translateGPUModuleToCubinAnnotation(llvm::Module &llvmModule,
                                                 Location loc, StringRef name);

  CubinGenerator cubinGenerator;
};

inline void emit_cuda_error(const llvm::Twine &message, const char *buffer,
                            CUresult error, Location loc) {
  emitError(loc, message.concat(" failed with error code ")
                     .concat(llvm::Twine{error})
                     .concat("[")
                     .concat(buffer)
                     .concat("]"));
}

#define RETURN_ON_CUDA_ERROR(expr, msg)                                        \
  {                                                                            \
    auto _cuda_error = (expr);                                                 \
    if (_cuda_error != CUDA_SUCCESS) {                                         \
      emit_cuda_error(msg, jitErrorBuffer, _cuda_error, loc);                  \
      return {};                                                               \
    }                                                                          \
  }

} // anonymous namespace

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

OwnedCubin KernelToBinaryPass::compilePtxToCubin(const std::string &ptx,
                                                 Location loc, StringRef name) {
  char jitErrorBuffer[4096] = {0};

  RETURN_ON_CUDA_ERROR(cuInit(0), "cuInit");

  // Linking requires a device context.
  CUdevice device;
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context;
  RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device), "cuCtxCreate");
  CUlinkState linkState;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *jitOptionsVals[] = {jitErrorBuffer,
                            reinterpret_cast<void *>(sizeof(jitErrorBuffer))};

  RETURN_ON_CUDA_ERROR(cuLinkCreate(2,              /* number of jit options */
                                    jitOptions,     /* jit options */
                                    jitOptionsVals, /* jit option values */
                                    &linkState),
                       "cuLinkCreate");

  RETURN_ON_CUDA_ERROR(
      cuLinkAddData(linkState, CUjitInputType::CU_JIT_INPUT_PTX,
                    const_cast<void *>(static_cast<const void *>(ptx.c_str())),
                    ptx.length(), name.data(), /* kernel name */
                    0,                         /* number of jit options */
                    nullptr,                   /* jit options */
                    nullptr                    /* jit option values */
                    ),
      "cuLinkAddData");

  void *cubinData;
  size_t cubinSize;
  RETURN_ON_CUDA_ERROR(cuLinkComplete(linkState, &cubinData, &cubinSize),
                       "cuLinkComplete");

  char *cubinAsChar = static_cast<char *>(cubinData);
  OwnedCubin result =
      std::make_unique<std::vector<char>>(cubinAsChar, cubinAsChar + cubinSize);

  // This will also destroy the cubin data.
  RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState), "cuLinkDestroy");
  return result;
}

OwnedCubin KernelToBinaryPass::convertModuleToCubin(llvm::Module &llvmModule,
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

  // Set the data layout of the llvm module to match what the ptx target needs.
  llvmModule.setDataLayout(targetMachine->createDataLayout());

  auto ptx = translateModuleToPtx(llvmModule, *targetMachine);

  return compilePtxToCubin(ptx, loc, name);
}

StringAttr KernelToBinaryPass::translateGPUModuleToCubinAnnotation(
    llvm::Module &llvmModule, Location loc, StringRef name) {
  auto cubin = convertModuleToCubin(llvmModule, loc, name);
  if (!cubin)
    return {};
  return StringAttr::get({cubin->data(), cubin->size()}, loc->getContext());
}

static PassRegistration<KernelToBinaryPass>
    pass("kernel-to-binary",
         "Convert all kernel functions to CUDA cubin blobs");
