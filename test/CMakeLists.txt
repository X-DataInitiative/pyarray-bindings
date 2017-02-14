enable_testing()

find_package(SWIG REQUIRED)
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

set(TICK_TEST_DIR ${PROJECT_SOURCE_DIR}/test)

add_executable(tick_array_test ${TICK_TEST_DIR}/array/test_array.cpp)

target_link_libraries(tick_array_test ${PYTHON_LIBRARIES} ${GTEST_BOTH_LIBRARIES} tick_array)
target_compile_definitions(tick_array_test PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION NO_IMPORT_ARRAY)

add_test(all_tick_tests tick_array_test)

add_subdirectory(python)