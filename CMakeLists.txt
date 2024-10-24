cmake_minimum_required(VERSION 3.15...3.29)
project(cgeneo LANGUAGES CXX)

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

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(cgeneo
    geneo/lib/create.cpp geneo/lib/output.cpp
    geneo/lib/identify.cpp geneo/lib/extract.cpp
    geneo/lib/describe.cpp geneo/lib/compute.cpp
    geneo/lib/cgeneo.cpp)

# Find OpenMP compiler flags
find_package(OpenMP COMPONENTS CXX)

# Add OpenMP flags if found
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Wall -Wextra")
  target_link_libraries(cgeneo PRIVATE OpenMP::OpenMP_CXX)
else()
  set(CMAKE_CXX_FLAGS "-Wall -Wextra")
endif()

install(TARGETS cgeneo DESTINATION .)