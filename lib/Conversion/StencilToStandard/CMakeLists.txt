add_llvm_library(MLIRStencilToStandard
        ConvertStencilToStandard.cpp
        )
add_dependencies(
        MLIRStencilToStandard

        MLIRAffineOps
        MLIRStencil
        MLIRStandardOps
        MLIRTransforms
        LLVMCore
        LLVMSupport
)
target_link_libraries(
        MLIRStencilToStandard

        MLIRAffineOps
        MLIRStandardOps
        MLIRStencil
        MLIRTransforms
        LLVMCore
        LLVMSupport
)

# specify header directories
target_include_directories(MLIRStencil PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(MLIRStencil PUBLIC "${PROJECT_BINARY_DIR}/include")