# The Open Earth Compiler

Development repository for the Open Earth Compiler. The compiler implements a stencil dialect and transformations that lower stencil programs to efficient GPU code. 

## Publication

A detailed discussion of the Open Earth Compiler can be found here:

[Domain-Specific Multi-Level IR Rewriting for GPU](https://arxiv.org/abs/2005.13014)

## Build Instructions

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-oec-opt
```
The ROCM_BACKEND_ENABLED flag enables the support for AMDGPU targets. It requires an LLVM build including lld and we need to set the path to lld using the following flag:
```sh
-DLLD_DIR=$PREFIX/lib/cmake/lld
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

## LLVM Build Instructions

The repository depends on a build of LLVM including MLIR. The Open Earth Compiler build has been tested with LLVM commit 54fafd17a72 using the following configuration:
```
cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DCMAKE_INSTALL_PREFIX=<install_root> -DLLVM_ENABLE_PROJECTS='mlir;lld' -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_INSTALL_UTILS=ON -DCMAKE_LINKER=<path_to_lld> -DLLVM_PARALLEL_LINK_JOBS=2
```
**Note**: Apply all patches found in the patch folder using git apply:
```
git apply ../stencil-dialect/patches/runtime.patch
```

## Compiling an Example Stencil Program

The following command lowers the laplace example stencil to NVIDIA GPU code:
```
oec-opt --stencil-shape-inference --convert-stencil-to-std --cse --parallel-loop-tiling='parallel-loop-tile-sizes=128,1,1' --canonicalize --test-gpu-greedy-parallel-loop-mapping --convert-parallel-loops-to-gpu --canonicalize --lower-affine --convert-scf-to-std --stencil-kernel-to-cubin ../test/Examples/laplace.mlir > laplace_lowered.mlir
```
**NOTE**: Use the command line flag --stencil-kernel-to-hsaco for AMD GPUs.

The tools mlir-translate and llc then convert the lowered code to an assembly file and/or object file:
```
mlir-translate --mlir-to-llvmir laplace_lowered.mlir > laplace.bc
llc -O3 laplace.bc -o laplace.s
clang -c laplace.s -o laplace.o
```
The generated object then exports the following method:
```
void _mlir_ciface_laplace(MemRefType3D *input, MemRefType3D *output);
```

