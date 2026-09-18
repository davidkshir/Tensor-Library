cmake_minimum_required(VERSION 4.3)
project(TensorLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(TensorLib
        src/TensorLayout.cpp
)

target_include_directories(TensorLib PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

#add_executable(TensorTest
#)
#
#target_link_libraries(TensorTest PRIVATE TensorLib
#)