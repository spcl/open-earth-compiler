# stencil-dialect

Development repository for the open earth compiler. The repository depends on a build of llvm including mlir. Before building mlir register your custom dialects in include/mlir/IR/DialectSymbolRegistry.def and change the main cmake file to install the td and def files. Once the llvm and mlir are built setup configure the project using the following commands.

```
mkdir build && cd build
cmake .. -DLLVM_DIR=<install_root>/lib/cmake/llvm/ -DLLVM_EXTERNAL_LIT=<build_root>/bin/llvm-lit
cmake --build . --target oec-opt
cmake --build . --target check-oec
```

# mlir main repo patches

In DialectSymbolRegistry.def:

```
 DEFINE_SYM_KIND_RANGE(SPIRV) // SPIR-V dialect
 DEFINE_SYM_KIND_RANGE(XLA_HLO) // XLA HLO dialect
 
+DEFINE_SYM_KIND_RANGE(STENCIL)
```

In the main CMakeLists.txt:
``` 
     FILES_MATCHING
     PATTERN "*.h"
     PATTERN "*.inc"
+    PATTERN "*.td"
+    PATTERN "*.def"
     PATTERN "LICENSE.TXT"
     )
```
