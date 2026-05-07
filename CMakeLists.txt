cmake_minimum_required(VERSION 3.16)
project(DecisionTree VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main executable
add_executable(decision_tree
    src/main.cpp
)

target_include_directories(decision_tree PRIVATE include)

# Compiler warnings
target_compile_options(decision_tree PRIVATE -Wall -Wextra -O2)

# Tests (optional but recommended)
enable_testing()
add_executable(run_tests tests/test_tree.cpp)
target_include_directories(run_tests PRIVATE include)
add_test(NAME TreeTests COMMAND run_tests)
