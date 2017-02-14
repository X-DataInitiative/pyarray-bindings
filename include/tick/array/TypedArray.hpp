#pragma once

#include <functional>
#include <iostream>
#include <tick/array/PyArray.hpp>

namespace tick {

struct FileLoadOptions {
    std::string sep = " ";
    std::size_t num;
};

namespace array_storage {

template <typename T>
struct Dense {
    PyArray<T> values;

    using is_sparse = std::false_type;
};

template <typename T>
struct SparseCRS {
    PyArray<T> values;
    PyArray<long> indices;
    PyArray<long> rows;

    using is_sparse = std::true_type;
};

}

template <typename T, typename Storage = array_storage::Dense<T>>
class TypedArray {
public:
    using Traits = typename detail::NpTypeTraits<T>;
    using CType = typename Traits::CType;

    TypedArray();
    TypedArray(PyRef&& data);

    TypedArray(npy_intp dim1);
    TypedArray(npy_intp dim1, npy_intp dim2);
    TypedArray(npy_intp dim1, npy_intp dim2, npy_intp dim3);

    TypedArray(const TypedArray& other);
    TypedArray(TypedArray&& other);

    TypedArray& operator=(const TypedArray& other);
    TypedArray& operator=(TypedArray&& other);

    DimensionsType Dimensions() const;

    double Mean() const;

    CType Sum() const;

    CType Min() const;
    CType Max() const;

    T& operator()(npy_intp idx1);
    T& operator()(npy_intp idx1, npy_intp idx2);
    T& operator()(npy_intp idx1, npy_intp idx2, npy_intp idx3);

    const T& operator()(npy_intp idx1) const;
    const T& operator()(npy_intp idx1, npy_intp idx2) const;
    const T& operator()(npy_intp idx1, npy_intp idx2, npy_intp idx3) const;

    template <typename OtherStorage>
    CType Dot(const TypedArray<T, OtherStorage>& other);

    TypedArray& Fill(const CType& value);
    TypedArray& Generate(std::function<T()> g);

    static TypedArray FromFile(const std::string& path, FileLoadOptions options = FileLoadOptions{});

    PyArray<T>& Values();
    const PyArray<T>& Values() const;

    Storage data;
};

using ArrayDouble       = TypedArray<double, array_storage::Dense<double>>;
using ArrayDoubleSparse = TypedArray<double, array_storage::SparseCRS<double>>;
using ArrayLong         = TypedArray<long, array_storage::Dense<long>>;
using ArrayLongSparse   = TypedArray<long, array_storage::SparseCRS<long>>;

}

#include "TypedArray.tpp"

