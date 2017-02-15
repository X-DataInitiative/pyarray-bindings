#define NO_IMPORT_ARRAY
#include "interface.hpp"

long example_array_long(tick::ArrayDouble & arr)
{
    arr.Fill(42);

    return arr.Sum();
}

long example_array_dot(tick::ArrayDouble &arr, const tick::ArrayDouble &arr2) {
    arr.Fill(4.0);

    return 5;
    //return arr.Dot(arr2);
}

extern tick::ArrayDouble example_array_return() {
    tick::ArrayDouble arr(10);

    arr.Fill(4.42);

    return arr;
}
