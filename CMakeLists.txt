cmake_minimum_required(VERSION 3.20)
project(cpp_drives_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python (use your venv via -DPython3_ROOT_DIR)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# pybind11 (installed via pip; CMake will find it using -DCMAKE_PREFIX_PATH)
find_package(pybind11 CONFIG REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE Python3::Python pybind11::embed)

# On Windows, ensure console subsystem (optional)
# set_target_properties(app PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
