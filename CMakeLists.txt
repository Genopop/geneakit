cmake_minimum_required(VERSION 3.15...3.29)
project(cgeneakit LANGUAGES CXX)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(cgeneakit
    geneakit/lib/create.cpp geneakit/lib/output.cpp
    geneakit/lib/identify.cpp geneakit/lib/extract.cpp
    geneakit/lib/describe.cpp geneakit/lib/compute.cpp
    geneakit/lib/edit.cpp geneakit/lib/cgeneakit.cpp)

# Find OpenMP. Apple Clang ships without it: use Homebrew's libomp if installed.
if(APPLE AND NOT DEFINED OpenMP_ROOT)
  execute_process(COMMAND brew --prefix libomp
                  OUTPUT_VARIABLE OpenMP_ROOT
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
endif()
find_package(OpenMP COMPONENTS CXX)

# The imported target carries the compile flags, include directory and library
if(OpenMP_CXX_FOUND)
  target_link_libraries(cgeneakit PRIVATE OpenMP::OpenMP_CXX)
else()
  message(STATUS "OpenMP not found: geneakit.phi() will run single-threaded")
endif()

if(MSVC)
  target_compile_options(cgeneakit PRIVATE /O2 /W4 /permissive-)
else()
  target_compile_options(cgeneakit PRIVATE -O3 -Wall -Wextra)
endif()

install(TARGETS cgeneakit DESTINATION .)
