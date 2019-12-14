cmake_minimum_required(VERSION 3.12.4)

project(oec-opt LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIR})

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

function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV} "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}" "-I${PROJECT_SOURCE_DIR}/include")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

function(whole_archive_link target)
   if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
     set(link_flags "-L${CMAKE_BINARY_DIR}/lib ")
     FOREACH(LIB ${ARGN})
       string(CONCAT link_flags ${link_flags} "-Wl,-force_load ${CMAKE_BINARY_DIR}/lib/lib${LIB}.a ")
     ENDFOREACH(LIB)
   elseif(MSVC)
     FOREACH(LIB ${ARGN})
       string(CONCAT link_flags ${link_flags} "/WHOLEARCHIVE:${LIB} ")
     ENDFOREACH(LIB)
   else()
    set(link_flags "-L${LLVM_LIBRARY_DIR} -Wl,--whole-archive,")
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "-l${LIB},")
    ENDFOREACH(LIB)
    string(CONCAT link_flags ${link_flags} "--no-whole-archive")
   endif()
  set_target_properties(${target} PROPERTIES LINK_FLAGS ${link_flags})
endfunction(whole_archive_link)

add_subdirectory(include/sten)
add_subdirectory(lib)
add_subdirectory(tools/oec-opt)
