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

# Link to external/include/parallel_hashmap/phmap.h
include_directories(${CMAKE_SOURCE_DIR}/external)

# Include Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

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
    geneakit/lib/cgeneakit.cpp)

# Find OpenMP compiler flags
find_package(OpenMP COMPONENTS CXX)

# Add OpenMP flags if found
if(OpenMP_FOUND)
  if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} /O3 /Wall /permissive-")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -Wall -Wextra")
  endif()
  target_link_libraries(cgeneakit PRIVATE OpenMP::OpenMP_CXX)
else()
  if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} /O3 /Wall /permissive-")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -Wall -Wextra")
  endif()
endif()

target_link_libraries(cgeneakit PRIVATE Eigen3::Eigen)
install(TARGETS cgeneakit DESTINATION .)