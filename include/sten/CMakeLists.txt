# Dialect/Sten/CMakeLists.txt

set(LLVM_TARGET_DEFINITIONS StenOps.td)

# Generate the class interfaces
mlir_tablegen(StenOps.h.inc -gen-op-decls)
# Generate the actual implementation
mlir_tablegen(StenOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MLIRStenOpsIncGen)
