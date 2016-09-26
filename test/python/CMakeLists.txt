cmake_minimum_required(VERSION 3.2)
project(xdata_array_python)

find_package(SWIG REQUIRED)

include(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs 3)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${XDATA_INCLUDE_DIR}/../../test/python)

SET_SOURCE_FILES_PROPERTIES(typemap.i PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(typemap.i PROPERTIES SWIG_FLAGS "-includeall")
SWIG_ADD_MODULE(xdata_array_test python typemap.i interface.cpp)
SWIG_LINK_LIBRARIES(xdata_array_test ${PYTHON_LIBRARIES} xdata_array)

target_compile_definitions(_xdata_array_test PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION NO_IMPORT_ARRAY)
