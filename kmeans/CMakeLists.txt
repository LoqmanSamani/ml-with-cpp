cmake_minimum_required(VERSION 3.15)
project(kmeans)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(kmeans src/main.cpp)

# Link libraries
target_link_libraries(kmeans PRIVATE Eigen3::Eigen)
