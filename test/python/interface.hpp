#pragma once

#ifndef SWIG
#include <tick/array/PyNdArray.hpp>
#endif

extern long example_array_long(tick::ArrayDouble & arr);
extern long example_array_dot(tick::ArrayDouble & arr, const tick::ArrayDouble& arr2);

extern tick::ArrayDouble example_array_return();