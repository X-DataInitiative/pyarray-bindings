enable_testing()

find_package(SWIG REQUIRED)
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

set(XDATA_TEST_DIR ${PROJECT_SOURCE_DIR}/test)

add_executable(xdata_array_test ${XDATA_TEST_DIR}/array/test_array.cpp)

target_link_libraries(xdata_array_test ${PYTHON_LIBRARIES} ${GTEST_BOTH_LIBRARIES} xdata_array)
target_compile_definitions(xdata_array_test PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION NO_IMPORT_ARRAY)

add_test(all_xdata_tests xdata_array_test)

add_subdirectory(python)