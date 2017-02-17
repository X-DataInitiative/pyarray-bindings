#pragma once

#include <iostream>
#include <array>

#include <Python.h>

#include <tick/array/PyRef.hpp>
#include <tick/array/TypeTraits.hpp>
#include <tick/array/PyNdArray.hpp>
#include <tick/array/PyScalar.hpp>

#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

namespace tick {

template <typename T>
class Array2D: public NdArray<T, 2> {
public:
    using Base = NdArray<T, 2>;
    using Traits = typename Base::Traits;
    using CType = typename Base::CType;

    Array2D()
        : Base(std::array<std::size_t, 2>{0, 0}) {
    }

    explicit Array2D(std::size_t x, std::size_t y)
        : Base(std::array<std::size_t, 2>{x, y}) {
    }

    explicit Array2D(PyRef&& other)
        : Base(std::move(other)) {
    }

    CType & operator()(std::size_t i, std::size_t j) {
        const auto dims = Base::Dimensions();

        return Base::at(i * dims[0] + j);
    }
};

using ArrayDouble2d = Array2D<double>;
using ArrayLong2d = Array2D<long>;

}  // namespace tick
