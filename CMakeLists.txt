cmake_minimum_required(VERSION 3.22)
project(stft C CXX)
find_package(GTest CONFIG REQUIRED)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

include_directories(gtest/include)
add_subdirectory(kiss_fft)
add_subdirectory(signal_processing)

# Main executable.
add_library(
        ${PROJECT_NAME}
        stft.c
        stft.h
)

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE C)

target_link_libraries(${PROJECT_NAME}
        kissfft
        signal_processing
)

# Add tests.
set(UnitTests ${PROJECT_NAME}_unit_tests)

add_executable(
        ${UnitTests}
        stft_test.cpp
)

set_target_properties(${UnitTests} PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(${UnitTests}
        GTest::gtest_main
        ${PROJECT_NAME}
)

include(GoogleTest)
gtest_discover_tests(${UnitTests})