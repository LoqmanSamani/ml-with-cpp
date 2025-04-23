cmake_minimum_required(VERSION 3.15)
project(lr)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen library
find_package(Eigen3 REQUIRED NO_MODULE)

# Add executable
add_executable(lr src/main.cpp)

# Link Eigen to the project
target_link_libraries(lr PRIVATE Eigen3::Eigen)