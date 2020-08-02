get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
)

add_llvm_executable(oec-translate oec-translate.cpp)

llvm_update_compile_flags(oec-translate)
target_link_libraries(oec-translate PRIVATE ${LIBS})
