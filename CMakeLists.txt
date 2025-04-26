cmake_minimum_required(VERSION 3.18)
project(gpu_dbms LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75) # Adjust for your GPU architecture

# Find required packages
find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)

# Include external dependencies
add_subdirectory(extern/sql-parser)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/sql-parser/src
)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)