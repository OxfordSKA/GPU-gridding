#! CMake initialisation.
cmake_minimum_required(VERSION 3.1)
if (POLICY CMP0045)
    cmake_policy(SET CMP0045 NEW)
endif()
if (POLICY CMP0068)
    cmake_policy(SET CMP0068 OLD)
endif()

#! Declare the CMake project.
project(grid_visibilities C CXX)

#! Find Threads.
find_package(Threads REQUIRED)

#! Find CUDA.
find_package(CUDA)
if (CUDA_FOUND)
    add_definitions(-DOSKAR_HAVE_CUDA)
endif()

#! Compiler flags.
if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*"
            OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    # Enable all warnings.
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -std=c++11 -arch=sm_60 -Xptxas -v -O2 -g -restrict -use_fast_math -D_FORCE_INLINES -D_DEBUG -lineinfo
)

#! Build type detection.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    get_filename_component(dirname ${CMAKE_BINARY_DIR} NAME)
    if (${dirname} MATCHES "dbg" OR ${dirname} MATCHES "debug")
        set(CMAKE_BUILD_TYPE Debug)
    else()
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

#! Build cfitsio library.
add_subdirectory(cfitsio-3.41)

#! Build oskar_binary library.
add_subdirectory(utils/binary)

#! Build test visibility gridder.
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/utils)
include_directories(${PROJECT_SOURCE_DIR}/cfitsio-3.41)
set(read_vis_SRC
    # List of source files to compile.
    utils/check_value.c
    utils/oskar_thread.c
    utils/oskar_timer.c
    utils/oskar_grid_weights.c
    utils/read_kernel.c
    utils/read_vis.c
    utils/write_fits_cube.c
    gpu_support.cu
    oskar_grid_wproj.c
    oskar_grid_wproj_gpu.cu
    main.cpp
)
if (CUDA_FOUND)
    cuda_add_executable(main ${read_vis_SRC})
else()
    add_executable(main ${read_vis_SRC})
endif()
target_link_libraries(main cfitsio oskar_binary m Threads::Threads)
