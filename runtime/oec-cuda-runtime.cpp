#include <bits/stdint-intn.h>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"

namespace {
int32_t reportError(CUresult result, const char *where) {
  if (result != CUDA_SUCCESS) {
    std::cerr << "-> OEC-RT Error: CUDA failed with " << result << " in "
              << where << "\n";
  }
  return result;
}

int32_t reportError(cudaError_t result, const char *where) {
  if (result != cudaSuccess) {
    std::cerr << "-> OEC-RT Error: CUDA failed with " << result << " in "
              << where << "\n";
  }
  return result;
}
} // anonymous namespace

static std::vector<void *> paramBuffer;
static std::vector<CUmodule> moduleBuffer;
static std::vector<CUdeviceptr> temporaryBuffer;
static CUstream stream;

extern "C" int32_t oecInit() {
  CUcontext context;
  CUdevice device;
  int32_t err;
  err = reportError(cuInit(0), "Init");
  err = reportError(cuDeviceGet(&device, 0), "Init");
  err = reportError(cuCtxCreate(&context, CU_CTX_SCHED_SPIN, device), "Init");
  err = reportError(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "StreamCreate");
  return err;
}

extern "C" int32_t oecTeardown() {
  int32_t err;
  for (auto module : moduleBuffer)
    err = reportError(cuModuleUnload(module), "ModuleUnload");
  for (auto param : paramBuffer)
    free(param);
  for (auto temporary : temporaryBuffer)
    err = reportError(cuMemFree(temporary), "MemFree");
  return err;
}

extern "C" void *oecAllocTemporary(int32_t size) {
  CUdeviceptr devPtr;
  reportError(cuMemAlloc(&devPtr, size), "MemAlloc");
  temporaryBuffer.push_back(devPtr);
  return reinterpret_cast<void *>(devPtr);
}

extern "C" int32_t oecModuleLoad(void **module, void *data) {
  int32_t err;
  err =
      reportError(cuModuleLoadData(reinterpret_cast<CUmodule *>(module), data),
                  "ModuleLoad");
  moduleBuffer.push_back(reinterpret_cast<CUmodule>(*module));
  return err;
}

extern "C" int32_t oecModuleGetFunction(void **function, void *module,
                                        const char *name) {
  int32_t err;
  err = reportError(
      cuModuleGetFunction(reinterpret_cast<CUfunction *>(function),
                          reinterpret_cast<CUmodule>(module), name),
      "GetFunction");
  err = reportError(cudaFuncSetAttribute(*function,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxL1), "SettingCarveout");
  return err;
}

extern "C" int32_t oecLaunchKernel(void *function, int32_t gridX,
                                   int32_t gridY, int32_t gridZ,
                                   int32_t blockX, int32_t blockY,
                                   int32_t blockZ, void **params) {
  return reportError(cuLaunchKernel(reinterpret_cast<CUfunction>(function),
                                    gridX, gridY, gridZ, blockX, blockY, blockZ,
                                    0, stream, params, nullptr),
                     "LaunchKernel");
}

extern "C" int32_t oecStreamSynchronize() {
  return reportError(cuStreamSynchronize(stream), "StreamSync");
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
