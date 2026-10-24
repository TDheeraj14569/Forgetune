cmake_minimum_required(VERSION 3.20)
project(ForgeTune VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position independent code for shared library compatibility
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core library sources
set(FORGETUNE_SOURCES
    src/ir.cpp
    src/graph_verifier.cpp
    src/interpreter.cpp
    src/pass_manager.cpp
    src/optimizer.cpp
    src/fusion.cpp
    src/codegen.cpp
    src/json_io.cpp
    src/memory_planner.cpp
)

# Library target
add_library(forgetune_lib STATIC ${FORGETUNE_SOURCES})
target_include_directories(forgetune_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# CLI compiler engine executable
add_executable(forgetune_engine src/main.cpp)
target_link_libraries(forgetune_engine PRIVATE forgetune_lib)
set_target_properties(forgetune_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)

# C++ native test suite
add_executable(forgetune_test_suite tests/cpp/test_main.cpp)
target_link_libraries(forgetune_test_suite PRIVATE forgetune_lib)
set_target_properties(forgetune_test_suite PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)

# Pybind11 extension target (if pybind11 is discovered)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(_forgetune_cpp src/bindings.cpp ${FORGETUNE_SOURCES})
    target_link_libraries(_forgetune_cpp PRIVATE forgetune_lib)
    set_target_properties(_forgetune_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/forgetune"
    )
endif()

enable_testing()
add_test(NAME CppNativeTestSuite COMMAND forgetune_test_suite)
