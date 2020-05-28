 
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace {
int32_t reportError(hipError_t result, const char *where) {
  if (result != hipSuccess) {
    std::cerr << "-> OEC-RT Error: HIP failed with " << result << " in "
              << where << "\n";
  }
  return result;
}

// int32_t reportError(hipError_t result, const char *where) {
//   if (result != hipSuccess) {
//     std::cerr << "-> OEC-RT Error: CUDA failed with " << result << " in "
//               << where << "\n";
//   }
//   return result;
// }
} // anonymous namespace

static std::vector<void *> paramBuffer;
static std::vector<hipModule_t> moduleBuffer;
static std::vector<void *> temporaryBuffer;

extern "C" int32_t oecInit() {
  // hipCtx_t context;
  hipDevice_t device;
  int32_t err = 0;
  err = reportError(hipInit(0), "Init");
  //err = reportError(hipDeviceGet(&device, 0), "Init");
  // err = reportError(hipCtxCreate(&context, hipDeviceScheduleSpin, device), "Init");
  // err = reportError(hipStreamCreate(&stream, hipStreamDefault), "StreamCreate");
  return err;
}

extern "C" int32_t oecTeardown() {
  int32_t err;
  for (auto module : moduleBuffer)
    err = reportError(hipModuleUnload(module), "ModuleUnload");
  for (auto param : paramBuffer)
    free(param);
  for (auto temporary : temporaryBuffer)
    err = reportError(hipFree(temporary), "MemFree");
  return err;
}

extern "C" void *oecAllocTemporary(int64_t size) {
  void* devPtr;
  reportError(hipMalloc(&devPtr, size), "MemAlloc");
  temporaryBuffer.push_back(devPtr);
  return reinterpret_cast<void *>(devPtr);
}

extern "C" int32_t oecModuleLoad(void **module, void *data) {
  int32_t err;
  err =
      reportError(hipModuleLoadData(reinterpret_cast<hipModule_t *>(module), data),
                  "ModuleLoad");
  moduleBuffer.push_back(reinterpret_cast<hipModule_t>(*module));
  return err;
}

extern "C" int32_t oecModuleGetFunction(void **function, void *module,
                                        const char *name) {
  int32_t err;
  err = reportError(
      hipModuleGetFunction(reinterpret_cast<hipFunction_t *>(function),
                          reinterpret_cast<hipModule_t>(module), name),
      "GetFunction");
  // err = reportError(cudaFuncSetAttribute(*function,
  //     cudaFuncAttributePreferredSharedMemoryCarveout,
  //     cudaSharedmemCarveoutMaxL1), "SettingCarveout");
  return err;
}

extern "C" int32_t oecLaunchKernel(void *function, intptr_t gridX,
                                   intptr_t gridY, intptr_t gridZ,
                                   intptr_t blockX, intptr_t blockY,
                                   intptr_t blockZ, void **params) {
  return reportError(hipModuleLaunchKernel(reinterpret_cast<hipFunction_t>(function),
                                    gridX, gridY, gridZ, blockX, blockY, blockZ,
                                    0, 0, params, nullptr),
                     "LaunchKernel");
}

extern "C" int32_t oecStreamSynchronize() {
  return reportError(hipDeviceSynchronize(), "StreamSync");
}

extern "C" void oecStoreParameter(void *paramPtr, int64_t size) {
  void *ptrToPtr = malloc(size);
  memcpy(ptrToPtr, paramPtr, size);
  paramBuffer.push_back(ptrToPtr);
}

extern "C" void oecLoadParameters(void **paramArray, int32_t offset,
                                  int32_t size) {
  for (size_t i = 0; i != size; ++i)
    paramArray[i] = reinterpret_cast<void *>(paramBuffer[offset + i]);
}
