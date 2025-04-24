cmake_minimum_required(VERSION 3.15)
project(lr)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(lr src/main.cpp)

# Include directories
target_include_directories(lr PRIVATE src/include)

# Link libraries
target_link_libraries(lr PRIVATE Eigen3::Eigen)