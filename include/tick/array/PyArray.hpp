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
class Array: public NdArray<T, 1> {
public:
    using Base = NdArray<T, 1>;
    using Traits = typename Base::Traits;
    using CType = typename Base::CType;

    Array()
        : Base(std::array<std::size_t, 1>{0}) {
    }

    explicit Array(std::size_t size)
            : Base(std::array<std::size_t, 1>{size}) {
    }

    explicit Array(PyRef&& other)
            : Base(std::move(other)) {
    }

    CType & operator()(std::size_t i) {
        return Base::at(i);
    }

    CType Dot(const Array<T>& other) {
        PyScalar<CType> scalar{
            PyRef{PyArray_InnerProduct(Base::GetPyRef().PyObj(), other.GetPyRef().PyObj())}
        };

        return scalar();
    }

//    PyArray Transposed() const;
//    PyArray Flattened() const;
//    PyArray FlattenedNoCopy() const;
};

using ArrayDouble = Array<double>;
using ArrayLong = Array<long>;

}  // namespace tick

//#include <tick/array/PyArray.tpp>
