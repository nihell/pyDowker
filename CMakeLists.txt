cmake_minimum_required(VERSION 3.15)
project(pyDowker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(_dowker_complex src/pyDowker/_dowker_complex.cpp)

install(TARGETS _dowker_complex DESTINATION pyDowker)
