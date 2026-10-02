cmake_minimum_required(VERSION 3.24)
project(BibitesGpuForkNative LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_library(BibitesGpuNative SHARED
    native/src/bgf_cuda.cu
    native/src/bgf_world.cu
    native/src/bgf_unity_interop.cpp
    native/include/bgf_api.h
    native/include/bgf_world.h)
target_include_directories(BibitesGpuNative PUBLIC native/include)
target_link_libraries(BibitesGpuNative PRIVATE d3d11)
set_target_properties(BibitesGpuNative PROPERTIES
    CUDA_ARCHITECTURES "75;86;89;120"
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION ON)

add_executable(BibitesGpuNativeTests native/tests/bgf_cuda_tests.cpp)
target_include_directories(BibitesGpuNativeTests PRIVATE native/include)
target_link_libraries(BibitesGpuNativeTests PRIVATE BibitesGpuNative)

add_executable(BibitesGpuBenchmark native/tests/bgf_benchmark.cpp)
target_include_directories(BibitesGpuBenchmark PRIVATE native/include)
target_link_libraries(BibitesGpuBenchmark PRIVATE BibitesGpuNative)

add_executable(BibitesGpuWorldTests native/tests/bgf_world_tests.cpp)
target_include_directories(BibitesGpuWorldTests PRIVATE native/include)
target_link_libraries(BibitesGpuWorldTests PRIVATE BibitesGpuNative d3d11)
target_compile_definitions(BibitesGpuWorldTests PRIVATE NOMINMAX)

add_executable(BibitesGpuEngine native/apps/bgf_world_app.cpp)
target_include_directories(BibitesGpuEngine PRIVATE native/include)
target_link_libraries(BibitesGpuEngine PRIVATE BibitesGpuNative)
set_target_properties(BibitesGpuEngine PROPERTIES OUTPUT_NAME "Bibites GPU Engine")

add_executable(BibitesGpuWorld WIN32 native/apps/bgf_world_viewer.cpp)
target_include_directories(BibitesGpuWorld PRIVATE native/include)
target_compile_definitions(BibitesGpuWorld PRIVATE UNICODE _UNICODE NOMINMAX)
target_link_libraries(BibitesGpuWorld PRIVATE BibitesGpuNative user32 gdi32 comctl32)
set_target_properties(BibitesGpuWorld PROPERTIES OUTPUT_NAME "Bibites GPU World")

enable_testing()
add_test(NAME cuda_brain_parity COMMAND BibitesGpuNativeTests)
add_test(NAME cuda_world_invariants COMMAND BibitesGpuWorldTests)
