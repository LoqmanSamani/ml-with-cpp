cmake_minimum_required(VERSION 3.10)
project(lr)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(lr main.cpp)
target_link_libraries(lr PRIVATE Eigen3::Eigen)
