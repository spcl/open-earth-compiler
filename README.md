# stencil-dialect

Development repository for the open earth compiler. Buld llvm and mlir in a separate folder and generate a build folder using the following commands:

```
mkdir build && cd build
cmake .. -DLLVM_DIR=<install_root>/lib/cmake/llvm/
cmake --build . --target occ-opt
```

Just for fun, the mlir-opt is called occ-opt here.
