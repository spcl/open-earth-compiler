# mlir_tutorial

- Build llvm with mlir

tested with
- llvm commit 40dfc6dff10bd8881c6df31884e2184bbaab5698
- mlir commit 0eee0ba2e75c7c5f1430621aa13d2f0367f297ad

needs a small tweek to the CMakeLists.txt of mlir
- install also *.td files and *.def files

Build this repository in the standard way, pointing to the LLVM installation

```
mkdir build && cd build
cmake .. -DLLVM_DIR=<install_root>/lib/cmake/llvm/
cmake --build . --target occ-opt
```

Just for fun, the mlir-opt is called occ-opt here.
