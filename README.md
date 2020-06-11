# The Stencil Dialect

Development repository for the open earth compiler. The repository depends on a build of llvm including mlir. The OEC build has been tested with LLVM commit c9f63297e24. 


## Build Instructions

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-oec-opt
```
The ROCM_BACKEND_ENABLED flag enables the support for AMDGPU tragets. It requires an llvm build with lld and we need to set the path to lld using the following flag:
```sh
-DLLD_DIR=$PREFIX/lib/cmake/lld
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

# LLVM Build Instructions

Cmake configuration for llvm
```
cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DCMAKE_INSTALL_PREFIX=<install_root> -DLLVM_ENABLE_PROJECTS='mlir;lld' -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_INSTALL_UTILS=ON -DMLIR_CUDA_RUNNER_ENABLED=ON -DCMAKE_CUDA_COMPILER=<path_to_nvcc> -DCMAKE_LINKER=<path_to_lld> -DLLVM_PARALLEL_LINK_JOBS=2
```
Do not forget to apply possible patches to llvm before compiling (patches located in stencil-dialect/patches).
