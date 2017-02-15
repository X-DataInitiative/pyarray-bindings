
%typemap(in) tick::ArrayDouble & (tick::ArrayDouble temp) {
    temp = tick::ArrayDouble{tick::PyRef::Take($input)};

    $1 = &temp;
}

%typemap(out) tick::ArrayDouble {
    tick::PyRef ref = std::move($1.GetPyRef());
    $result = ref.PyObj();
    Py_XINCREF($result);
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
