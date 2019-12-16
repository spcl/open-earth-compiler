# get the source directory
if (LLVM_BUILD_MAIN_SRC_DIR)
  set(LLVM_SOURCE_DIR ${LLVM_BUILD_MAIN_SRC_DIR} CACHE PATH
    "Path to LLVM source tree")
else()
  execute_process(COMMAND "${LLVM_TOOLS_BINARY_DIR}/llvm-config" --src-root
    OUTPUT_VARIABLE MAIN_SRC_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(LLVM_SOURCE_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")
endif()

#find_program(LLVM_FILECHECK NAMES FileCheck HINTS ${LLVM_SOURCE_DIR}/../build/bin)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(OEC_TEST_DEPENDS
  oec-opt
  )

add_lit_testsuite(check-oec "Running the OEC regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${OEC_TEST_DEPENDS}
  )
set_target_properties(check-oec PROPERTIES FOLDER "Tests")

#add_lit_testsuites(MLIR ${CMAKE_CURRENT_SOURCE_DIR}
#  DEPENDS ${OEC_TEST_DEPENDS}
#)
