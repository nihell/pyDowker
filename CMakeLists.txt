cmake_minimum_required(VERSION 3.18)
project(dowker_complex LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_dowker_complex src/pyDowker/_dowker_complex.cpp)

install(TARGETS _dowker_complex DESTINATION pyDowker)
