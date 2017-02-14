
%typemap(in) tick::ArrayDouble & (tick::ArrayDouble temp) {
    Py_XINCREF($input);

    temp = tick::ArrayDouble{tick::PyRef{std::move($input)}};

    // PyRef::From(PyObject*&);
    // Inc, set to null;

    $input = nullptr;

    $1 = &temp;
}

%init %{
    tick::PyRef::Init();
%}

%module tick_array_mod
%{

#include <tick/array/PyNdArray.hpp>
#include "interface.hpp"

%}

#include "interface.hpp"
