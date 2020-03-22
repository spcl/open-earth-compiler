# stencil-dialect

Development repository for the open earth compiler. The repository depends on a build of llvm including mlir. Before building mlir register your custom dialects in include/mlir/IR/DialectSymbolRegistry.def and change the main cmake file to install the td and def files. Once the llvm and mlir are built setup configure the project using the following commands.

```
mkdir build && cd build
cmake -G Ninja .. -DCMAKE_LINKER=<path_to_lld> -DLLVM_DIR=<install_root>/lib/cmake/llvm/ -DLLVM_EXTERNAL_LIT=<build_root>/bin/llvm-lit
cmake --build . --target oec-opt
cmake --build . --target check-oec
```

# llvm build instructions

Cmake configuration for llvm

```
cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="host;NVPTX" -DCMAKE_INSTALL_PREFIX=<install_root> -DLLVM_ENABLE_PROJECTS='mlir' -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_INSTALL_UTILS=ON -DMLIR_CUDA_RUNNER_ENABLED=ON -DCMAKE_CUDA_COMPILER=<path_to_nvcc> -DCMAKE_LINKER=<path_to_lld> -DLLVM_PARALLEL_LINK_JOBS=2
```

Do not forget to apply possible patches to llvm before compiling (patches located in stencil-dialect/patches).
