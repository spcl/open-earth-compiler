# stencil-dialect

Development repository for the open earth compiler. The repository depends on a build of llvm and mlir. Before building mlir  register your custom dialects in include/mlir/IR/DialectSymbolRegistry.def. Once the llvm and mlir are built setup configure the project using the following commands.

```
mkdir build && cd build
cmake .. -DLLVM_DIR=<install_root>/lib/cmake/llvm/ -DLLVM_EXTERNAL_LIT=<build_root>/bin/llvm-lit
cmake --build . --target oec-opt
cmake --build . --target check-oec
```
