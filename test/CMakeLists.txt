enable_testing()

find_package(SWIG REQUIRED)
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})


set(XDATA_TEST_DIR ${PROJECT_SOURCE_DIR}/test)



add_executable(xdata_array_test ${XDATA_TEST_DIR}/array/test_array.cpp python/interface.cpp python/interface.hpp)
target_link_libraries(xdata_array_test ${PYTHON_LIBRARIES} ${GTEST_BOTH_LIBRARIES} xdata_array)

add_test(all_xdata_tests xdata_array_test)

add_subdirectory(python)