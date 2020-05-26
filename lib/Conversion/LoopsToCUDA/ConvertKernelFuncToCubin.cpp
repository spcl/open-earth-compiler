#include "Conversion/LoopsToCUDA/Passes.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/ROCDLIR.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/TargetSelect.h"

#include "cuda.h"

using namespace mlir;

static LogicalResult initAMDGPUBackendCallback() {
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
  return success();
}

static LogicalResult
compileModuleToROCDLIR(Operation *m,
                       std::unique_ptr<llvm::Module> &llvmModule) {
  llvmModule = translateModuleToROCDLIR(m);
  if (llvmModule)
    return success();
  return failure();
}

static OwnedBlob compileIsaToHsaco(const std::string &input, Location,
                                             StringRef) {
  return std::make_unique<std::vector<char>>(input.begin(), input.end());
}

namespace mlir {
void registerGPUToCUBINPipeline() {
  PassPipelineRegistration<>(
      "stencil-gpu-to-cubin", "Lowering of stencil kernels to cubins",
      [](OpPassManager &pm) {
        pm.addPass(createGpuKernelOutliningPass());
        auto &kernelPm = pm.nest<gpu::GPUModuleOp>();
        kernelPm.addPass(createStripDebugInfoPass());
        kernelPm.addPass(createLowerGpuOpsToROCDLOpsPass());
        kernelPm.addPass(createStencilIndexOptimizationPass());
        kernelPm.addPass(createConvertGPUKernelToBlobPass(
            initAMDGPUBackendCallback, compileModuleToROCDLIR,
            compileIsaToHsaco, "amdgcn-amd-amdhsa", "gfx1010",
            "-code-object-v3", "rocdl.hsaco"));
        LowerToLLVMOptions llvmOptions;
        llvmOptions.emitCWrappers = true;
        llvmOptions.useAlignedAlloc = false;
        llvmOptions.useBarePtrCallConv = false;
        llvmOptions.indexBitwidth = kDeriveIndexBitwidthFromDataLayout;
        pm.addPass(createLowerToLLVMPass(llvmOptions));
      });
}
} // namespace mlir
