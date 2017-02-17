
%define TYPEMAP_ARRAY(ARRAY_TYPE)
%typemap(in) ARRAY_TYPE & (ARRAY_TYPE temp) {
    temp = ARRAY_TYPE{tick::PyRef::Take($input)};
    $1 = &temp;
}
%typemap(out) ARRAY_TYPE {
    tick::PyRef ref = std::move($1.GetPyRef());
    $result = ref.PyObj();
    Py_XINCREF($result);
}
%enddef

TYPEMAP_ARRAY(tick::ArrayDouble)
TYPEMAP_ARRAY(tick::ArrayLong)

%init %{
    tick::PyRef::Init();
%}

%module tick_array_mod
%{

#include <tick/array/PyNdArray.hpp>
#include "interface.hpp"

%}

#include "interface.hpp"
