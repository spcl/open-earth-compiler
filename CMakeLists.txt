cmake_minimum_required(VERSION 3.12.4)

project(oec-opt LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)

if(LLVM_FOUND)
  message(STATUS "LLVM found")
endif()

# MLIR project.
set(MLIR_MAIN_SRC_DIR ${LLVM_INCLUDE_DIR}) # --src-root
set(MLIR_INCLUDE_DIR ${LLVM_INCLUDE_DIR}) # --includedir
set(MLIR_TABLEGEN_EXE mlir-tblgen)

# import llvm functionality
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV} "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}" "-I${PROJECT_SOURCE_DIR}/include")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

function(whole_archive_link target)
  set(link_flags "-L${LLVM_LIBRARY_DIR} -Wl,--whole-archive,")
  FOREACH(LIB ${ARGN})
    string(CONCAT link_flags ${link_flags} "-l${LIB},")
  ENDFOREACH(LIB)
  string(CONCAT link_flags ${link_flags} "--no-whole-archive")
  set_target_properties(${target} PROPERTIES LINK_FLAGS ${link_flags})
endfunction(whole_archive_link)

add_subdirectory(include/)
add_subdirectory(lib/)
add_subdirectory(test/)
add_subdirectory(oec-opt/)
