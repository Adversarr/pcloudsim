cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

project(pcloudsim_binding LANGUAGES CXX CUDA)
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

include(cmake/CPM.cmake)

CPMAddPackage(
    NAME Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    SYSTEM TRUE
    OPTIONS
        "EIGEN_BUILD_DOC OFF"
)

find_package(OpenMP)

nanobind_add_module(pcloudsim_binding
  NB_STATIC STABLE_ABI NOSTRIP
  pcloudsim.cpp sampling.cpp bind.cpp)
target_link_libraries(pcloudsim_binding PUBLIC Eigen3::Eigen)
set_target_properties(pcloudsim_binding PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (OpenMP_CXX_FOUND)
    target_link_libraries(pcloudsim_binding PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(pcloudsim_binding PUBLIC PCLOUDSIM_HAS_OPENMP)
endif()

if (GNUC)
    target_compile_options(pcloudsim_binding PRIVATE -march=native)
endif()

# Install directive for scikit-build-core
install(TARGETS pcloudsim_binding LIBRARY DESTINATION pcloudsim)