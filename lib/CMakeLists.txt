# Dialect/Sten/CMakeLists.txt

add_llvm_library(MLIRSten
        DialectRegistration.cpp
        StenDialect.cpp
        StenOps.cpp
        StenTypes.cpp)

# Make sure that the TableGen generated files are up-to-date
add_dependencies(MLIRSten
        MLIRStenOpsIncGen)

target_link_libraries(MLIRSten
        MLIRIR
        MLIRParser
        MLIRSupport)

# specify header directories
target_include_directories(MLIRSten PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(MLIRSten PUBLIC ${PROJECT_BINARY_DIR}/include)
