#pragma once

#ifndef SWIG
#include <tick/array/PyArray.hpp>
#endif

extern long example_array_long(tick::ArrayLong & arr);
extern long example_array_dot(tick::ArrayDouble & arr, const tick::ArrayLong& arr2);
extern tick::ArrayDouble example_array_return();