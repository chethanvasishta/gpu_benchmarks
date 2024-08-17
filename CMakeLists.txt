cmake_minimum_required(VERSION 3.22)
project(gpu_benchmarks LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
find_package(CUDA REQUIRED)

add_executable(gpu_benchmarks source/main.cu source/driver.cpp)
set_target_properties(gpu_benchmarks PROPERTIES CUDA_ARCHITECTURES 75)

