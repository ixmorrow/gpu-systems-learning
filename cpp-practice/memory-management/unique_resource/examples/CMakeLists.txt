# Specify the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(matrix_multiply)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the source file
add_executable(matrix_multiply matrix_multiply.cpp)

# Include the parent directory (if unique_resource.h is there)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../include/)