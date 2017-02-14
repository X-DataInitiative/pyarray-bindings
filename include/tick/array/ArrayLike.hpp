#pragma once

#include <tuple>
#include <vector>

namespace tick {

template <typename T>
struct ArrayLike {
    std::tuple<std::size_t, std::size_t> dimensions;
    unsigned int ndims;

    virtual
    T* Data() = 0;

    virtual
    std::vector<T> AsStdVector() = 0;

};

}
