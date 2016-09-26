#pragma once

#include <iostream>
#include <array>

#include <xdata/array/PyRef.hpp>
#include <xdata/array/TypeTraits.hpp>
#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

namespace xdata {

using DimensionsType =  std::array<npy_intp, 3>;

template <typename T>
class PyArray {
public:
    using Traits = typename detail::NpTypeTraits<T>;
    using CType = typename Traits::CType;

    PyArray();
    PyArray(PyRef&& ref);

    PyArray(const PyArray& other);
    PyArray(PyArray&& other);

    PyArray& operator=(const PyArray& other);
    PyArray& operator=(PyArray&& other);

    PyArray(std::array<npy_intp, 1>);
    PyArray(std::array<npy_intp, 2>);
    PyArray(std::array<npy_intp, 3>);

    PyObject* PyObj() const;
    PyArrayObject* PyArrayObj() const;

    DimensionsType Dimensions() const;
    int NDims() const;

    bool IsWellFormed() const;

    CType& operator()(npy_intp idx1);
    CType& operator()(npy_intp idx1, npy_intp idx2);
    CType& operator()(npy_intp idx1, npy_intp idx2, npy_intp idx3);

    const CType& operator()(npy_intp idx1) const;
    const CType& operator()(npy_intp idx1, npy_intp idx2) const;
    const CType& operator()(npy_intp idx1, npy_intp idx2, npy_intp idx3) const;

//    PyArray Transposed() const;
//    PyArray Flattened() const;
//    PyArray FlattenedNoCopy() const;

    CType* Data() const;

private:
    PyRef data;
};

}

#include <xdata/array/PyArray.tpp>
