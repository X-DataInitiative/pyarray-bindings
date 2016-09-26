
%typemap(in) xdata::ArrayLong & (xdata::ArrayLong temp) {
    Py_XINCREF($input);

    temp = xdata::ArrayLong{std::move($input)};

    $1 = &temp;
}

%init %{
    xdata::PyRef::Init();
%}

%module xdata_array_test
%{

#include <xdata/array/TypedArray.hpp>
#include "interface.hpp"

%}

#include "interface.hpp"
