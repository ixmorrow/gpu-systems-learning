cmake_minimum_required(VERSION 3.14)
project(unique_resource)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include Google Test for testing
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Add the include directory
include_directories(include)

# Enable testing
enable_testing()
add_subdirectory(tests)