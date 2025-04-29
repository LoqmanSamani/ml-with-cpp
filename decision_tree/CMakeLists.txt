cmake_minimum_required(VERSION 3.15)
project(decision_tree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(dtree src/main.cpp)

# Link libraries
target_link_libraries(decision_tree PRIVATE Eigen3::Eigen)
