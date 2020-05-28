#include "Conversion/LoopsToCUDA/Passes.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/ROCDLIR.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/TargetSelect.h"

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;

static LogicalResult initAMDGPUBackendCallback() {
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUAsmParser();
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

static OwnedBlob compileIsaToHsaco(const std::string &input, Location loc,
                                   StringRef) {


  // Initialize the target
  std::string tripleName = "amdgcn-amd-amdhsa";
  std::string mCPU = "gfx906";
  std::string featureStr = "+code-object-v3,+sram-ecc"; 
  llvm::Triple triple(tripleName);
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  std::string error;
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget("", triple, error);
  if (TheTarget == nullptr) {
    emitError(loc, "cannot initialize target triple");
    return {};
  }

  // Setup the source manager
  llvm::SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(input),
                            llvm::SMLoc());

  std::unique_ptr<llvm::MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(tripleName));
  assert(MRI && "Unable to create target register info!");

  llvm::MCTargetOptions MCOptions;
  std::unique_ptr<llvm::MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, tripleName, MCOptions));
  assert(MAI && "Unable to create target asm info!");

  // Setup the context
  llvm::MCObjectFileInfo MOFI;
  llvm::MCContext Ctx(MAI.get(), MRI.get(), &MOFI, &SrcMgr, &MCOptions);
  MOFI.InitMCObjectFileInfo(triple, false, Ctx, false);

  std::unique_ptr<llvm::MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  std::unique_ptr<llvm::MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(tripleName, mCPU, featureStr));

  llvm::MCCodeEmitter *CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, Ctx);
  llvm::MCAsmBackend *MAB =
      TheTarget->createMCAsmBackend(*STI, *MRI, MCOptions);

  SmallString<0> Storage;
  Storage.clear();
  llvm::raw_svector_ostream OS(Storage);
  std::unique_ptr<llvm::MCStreamer> Str;

  Str.reset(TheTarget->createMCObjectStreamer(
      triple, Ctx, std::unique_ptr<llvm::MCAsmBackend>(MAB),
      MAB->createObjectWriter(OS), std::unique_ptr<llvm::MCCodeEmitter>(CE),
      *STI, MCOptions.MCRelaxAll, MCOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false));

  std::unique_ptr<llvm::MCAsmParser> Parser(llvm::createMCAsmParser(SrcMgr, Ctx, *Str, *MAI));
  std::unique_ptr<llvm::MCTargetAsmParser> TAP(
      TheTarget->createMCAsmParser(*STI, *Parser, *MCII, MCOptions));

  if (TAP == nullptr) {
    emitError(loc, "cannot initialize target asm parser");
    return {};
  }
  
  Parser->setTargetParser(*TAP);
  int Res = Parser->Run(true);

  if (Res != 0) {
    emitError(loc, "failed to assemble the stuff");
    return {};
  }
  return std::make_unique<std::vector<char>>(Storage.begin(), Storage.end());
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
            compileIsaToHsaco, "amdgcn-amd-amdhsa", "gfx906", "+code-object-v3,+sram-ecc",
            "rocdl.hsaco"));
        LowerToLLVMOptions llvmOptions;
        llvmOptions.emitCWrappers = true;
        llvmOptions.useAlignedAlloc = false;
        llvmOptions.useBarePtrCallConv = false;
        llvmOptions.indexBitwidth = kDeriveIndexBitwidthFromDataLayout;
        pm.addPass(createLowerToLLVMPass(llvmOptions));
      });
}
} // namespace mlir
