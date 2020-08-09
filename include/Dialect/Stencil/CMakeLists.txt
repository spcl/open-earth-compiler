add_mlir_dialect(StencilOps stencil)

set(LLVM_TARGET_DEFINITIONS StencilInterfaces.td)
mlir_tablegen(StencilOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(StencilOpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRStencilInterfacesIncGen)
add_dependencies(mlir-headers MLIRStencilInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Stencil)
add_public_tablegen_target(MLIRStencilPassIncGen)
add_dependencies(mlir-headers MLIRStencilPassIncGen)

add_mlir_doc(StencilDialect -gen-dialect-doc StencilDialect Stencil/)
add_mlir_doc(StencilInterfaces -gen-op-interface-doc StencilInterfaces Stencil/)
add_mlir_doc(StencilOps -gen-op-doc StencilOps Stencil/)