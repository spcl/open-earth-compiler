
llvm_map_components_to_libnames(nvptx "NVPTX")

add_llvm_library(MLIRKernelFuncToBinary
        ConvertKernelFuncToBinary.cpp
        )
add_dependencies(
        MLIRKernelFuncToBinary

        MLIRGPU
        MLIRLLVMIR
        MLIRNVVMIR
        MLIRPass
        MLIRTargetNVVMIR
        ${nvptx}
)
target_link_libraries(
        MLIRKernelFuncToBinary

        MLIRGPU
        MLIRLLVMIR
        MLIRNVVMIR
        MLIRPass
        MLIRTargetNVVMIR
        ${nvptx}
)
