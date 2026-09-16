cmake_minimum_required(VERSION 4.3)
project(TensorLib)

set(CMAKE_CXX_STANDARD 20)

add_library(TensorLib
        src/TensorLayout.cpp
)

target_link_libraries(TensorTests PRIVATE TensorLib)

target_include_directories(TensorLib PUBLIC
        include
        include/details
)