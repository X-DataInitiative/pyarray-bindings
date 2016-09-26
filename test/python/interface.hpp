#pragma once

#ifndef SWIG
#include <xdata/array/TypedArray.hpp>
#endif

extern long example_array_long(xdata::ArrayLong & arr);
extern long example_array_dot(xdata::ArrayLong & arr, const xdata::ArrayLong& arr2);