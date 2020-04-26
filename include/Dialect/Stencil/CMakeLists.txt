add_mlir_dialect(StencilOps stencil)
add_mlir_doc(StencilDialect -gen-dialect-doc StencilDialect Stencil/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRStencilPassIncGen)

add_mlir_doc(StencilOps -gen-op-doc StencilOps Stencil/)