# Dialect/Sten/CMakeLists.txt

add_library(MLIRSten
        DialectRegistration.cpp
        StenDialect.cpp
        StenOps.cpp
        StenTypes.cpp

        # ADDITIONAL_HEADER_DIRS
        # ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Sten
        )

# Make sure that the TableGen generated files are up-to-date
add_dependencies(MLIRSten
        MLIRStenOpsIncGen)

target_link_libraries(MLIRSten
        MLIRIR
        MLIRParser
        MLIRSupport)

target_include_directories(MLIRSten PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(MLIRSten PUBLIC ${PROJECT_BINARY_DIR}/include)

link_directories(${})
