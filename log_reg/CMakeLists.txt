cmake_minimum_required(VERSION 3.15)
project(log_reg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(log_reg src/main.cpp)

# Include directories
#target_include_directories(log_reg PRIVATE src/include)

# Link libraries
target_link_libraries(log_reg PRIVATE Eigen3::Eigen)
