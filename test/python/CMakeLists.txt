cmake_minimum_required(VERSION 2.8)
project(tick_array_python)

find_package(SWIG REQUIRED)

include(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs 3)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${TICK_INCLUDE_DIR}/../../test/python)

SET_SOURCE_FILES_PROPERTIES(typemap.i PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(typemap.i PROPERTIES SWIG_FLAGS "-includeall")
SWIG_ADD_MODULE(tick_array_mod python typemap.i interface.cpp)
SWIG_LINK_LIBRARIES(tick_array_mod ${PYTHON_LIBRARIES} tick_array)

target_compile_definitions(_tick_array_mod PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION NO_IMPORT_ARRAY)
