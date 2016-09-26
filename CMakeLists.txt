cmake_minimum_required(VERSION 3.2)
project(xdata_array)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMake/Modules)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(NumPy REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(BLAS)
find_package(Boost COMPONENTS system timer)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})

set(XDATA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/xdata)
set(XDATA_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/xdata)

set(XDATA_SOURCE_FILES
        ${XDATA_SOURCE_DIR}/array/PyRef.cpp
        ${XDATA_SOURCE_DIR}/array/TypedArray.cpp
        ${XDATA_SOURCE_DIR}/array/PyArray.cpp)

set(XDATA_INCLUDE_FILES
        ${XDATA_INCLUDE_DIR}/array/Fwd.hpp
        ${XDATA_INCLUDE_DIR}/array/TypeTraits.hpp
        ${XDATA_INCLUDE_DIR}/array/PyRef.hpp
        ${XDATA_INCLUDE_DIR}/array/PyScalar.tpp
        ${XDATA_INCLUDE_DIR}/array/PyScalar.hpp
        ${XDATA_INCLUDE_DIR}/array/PyArray.hpp
        ${XDATA_INCLUDE_DIR}/array/PyArray.tpp
        ${XDATA_INCLUDE_DIR}/array/ArrayLike.hpp
        ${XDATA_INCLUDE_DIR}/array/TypedArray.hpp
        ${XDATA_INCLUDE_DIR}/array/TypedArray.tpp
        )

add_library(xdata_array ${XDATA_SOURCE_FILES} ${XDATA_INCLUDE_FILES})

target_compile_definitions(xdata_array PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION PY_ARRAY_UNIQUE_SYMBOL=XDATA_PYARRAY_API)
target_link_libraries(xdata_array ${PYTHON_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES})

if (${BLAS_FOUND})
    message(STATUS "BLAS found, libraries: ${BLAS_LIBRARIES}")
    message(STATUS "BLAS found, dir: ${BLAS_LINKER_FLAGS}")
endif()

add_subdirectory(test)
