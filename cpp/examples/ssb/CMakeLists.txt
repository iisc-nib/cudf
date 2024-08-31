# Copyright (c) 2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

rapids_cuda_init_architectures(ssb_example)
rapids_cuda_set_architectures(RAPIDS)

project(
  ssb_example
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)


list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
include(/media/ajayakar/space/src/cudf/cpp/cmake/thirdparty/get_cucollections.cmake)

add_executable(q21-cuco q21-cuco.cu)
target_compile_options(q21-cuco PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(q21-cuco 
PRIVATE cuco::cuco )

add_executable(q21-thrust-cuco q21-thrust-cuco.cu)
target_compile_options(q21-thrust-cuco PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(q21-thrust-cuco 
PRIVATE cuco::cuco )
