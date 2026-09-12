cmake_minimum_required(VERSION 4.3)
project(TensorLib)

set(CMAKE_CXX_STANDARD 20)

add_executable(TensorLib
        include/Storage.hpp
        include/TensorLayout.hpp
        src/TensorLayout.cpp
)

target_include_directories(TensorLib PUBLIC include)