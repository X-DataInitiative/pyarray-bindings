cmake_minimum_required(VERSION 2.8)
project(tick_array)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMake/Modules)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(NumPy REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(BLAS)
find_package(Boost COMPONENTS system timer)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})

set(TICK_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/tick)
set(TICK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick)

set(TICK_SOURCE_FILES
        ${TICK_SOURCE_DIR}/array/PyRef.cpp)

set(TICK_INCLUDE_FILES
        ${TICK_INCLUDE_DIR}/array/Fwd.hpp
        ${TICK_INCLUDE_DIR}/array/TypeTraits.hpp
        ${TICK_INCLUDE_DIR}/array/PyRef.hpp
        ${TICK_INCLUDE_DIR}/array/PyScalar.tpp
        ${TICK_INCLUDE_DIR}/array/PyScalar.hpp
        ${TICK_INCLUDE_DIR}/array/PyArray.hpp
        ${TICK_INCLUDE_DIR}/array/PyArray.tpp
        ${TICK_INCLUDE_DIR}/array/ArrayLike.hpp
        ${TICK_INCLUDE_DIR}/array/TypedArray.hpp
        ${TICK_INCLUDE_DIR}/array/TypedArray.tpp
        )

add_library(tick_array ${TICK_SOURCE_FILES} ${TICK_INCLUDE_FILES})

set_property(TARGET tick_array PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(tick_array PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION PY_ARRAY_UNIQUE_SYMBOL=TICK_PYARRAY_API)
target_link_libraries(tick_array ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} -lpthread -lm)

if (${BLAS_FOUND})
    message(STATUS "BLAS found, libraries: ${BLAS_LIBRARIES}")
    message(STATUS "BLAS found, dir: ${BLAS_LINKER_FLAGS}")
endif()

add_subdirectory(test)
