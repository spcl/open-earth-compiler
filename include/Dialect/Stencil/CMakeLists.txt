set(LLVM_TARGET_DEFINITIONS StencilOps.td)

# Generate the class interfaces
mlir_tablegen(StencilOps.h.inc -gen-op-decls)
# Generate the actual implementation
mlir_tablegen(StencilOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MLIRStencilOpsIncGen)
