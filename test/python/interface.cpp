#define NO_IMPORT_ARRAY
#include "interface.hpp"

long example_array_long(tick::ArrayLong & arr) {
    arr.Fill(42);

    return arr.Sum();
}

long example_array_dot(tick::ArrayDouble &arr, const tick::ArrayLong &arr2) {
    arr.Fill(arr2.Sum());

    return 5;
}

tick::ArrayDouble example_array_return() {
    tick::ArrayDouble arr(10);

    arr.Fill(4.42);

    return arr;
}
