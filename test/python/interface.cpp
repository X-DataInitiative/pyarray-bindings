
#include "interface.hpp"


long example_array_long(xdata::ArrayLong & arr)
{
    arr.Fill(42);

    return arr.Sum();
}

long example_array_dot(xdata::ArrayLong &arr, const xdata::ArrayLong &arr2) {
    return arr.Dot(arr2);
}
