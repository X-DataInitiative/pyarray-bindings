#pragma once

#include <Python.h>
#include <numpy/arrayobject.h>

#include <xdata/array/PyScalar.hpp>
#include <xdata/array/TypedArray.hpp>

namespace xdata {

template <typename T, typename S, unsigned long N>
struct InitializeData {};

template <typename T, unsigned long N>
struct InitializeData<T, array_storage::Dense<T>, N> {
    array_storage::Dense<T> operator()(std::array<npy_intp, N> dims)
    {
        return array_storage::Dense<T>{dims};
    }
};

template <typename T, unsigned long N>
struct InitializeData<T, array_storage::SparseCRS<T>, N> {
    array_storage::SparseCRS<T> operator()(std::array<npy_intp, N> dims)
    {
        return array_storage::SparseCRS<T>{dims, std::array<npy_intp, 1>{0}, std::array<npy_intp, 1>{0}};
    }
};

template <typename T, typename S>
TypedArray<T, S>::TypedArray()
    : data{InitializeData<T, S, 1ul>{}({0})}
{
}

template <typename T, typename S>
TypedArray<T, S>::TypedArray(PyRef&& ref)
    : data{S{PyArray<T>{std::move(ref)}}}
{
}

template <typename T, typename S>
TypedArray<T, S>::TypedArray(npy_intp dim1)
    : data{InitializeData<T, S, 1ul>{}({dim1})}
{
}

template <typename T, typename S>
TypedArray<T, S>::TypedArray(npy_intp dim1, npy_intp dim2)
    : data{InitializeData<T, S, 2ul>{}({dim1, dim2})}
{
}

template <typename T, typename S>
TypedArray<T, S>::TypedArray(npy_intp dim1, npy_intp dim2, npy_intp dim3)
    : data{InitializeData<T, S, 3ul>{}({dim1, dim2, dim3})}
{
}

template <typename T, typename S>
TypedArray<T, S>& TypedArray<T, S>::Fill(const CType &value)
{
    PyScalar<CType> s{value};

    PyArray_FillWithScalar(data.values.PyArrayObj(), s.PyObj());

    return *this;
}

template <typename T, typename S>
TypedArray<T, S>& TypedArray<T, S>::Generate(std::function<T()> g)
{
    std::generate(data.values.Data(), data.values.Data() + PyArray_Size(data.values.PyObj()), std::move(g));

    return *this;
}

template <typename T, typename S>
typename TypedArray<T, S>::CType TypedArray<T, S>::Sum() const {
    PyScalar<T> s{PyArray_Sum(Values().PyArrayObj(), NPY_MAXDIMS, Traits::NpType, nullptr)};

    return s.Value();
}

template <typename T, typename S>
double TypedArray<T, S>::Mean() const {
    PyScalar<double> s{PyArray_Mean(data.values.PyArrayObj(), NPY_MAXDIMS, Traits::NpType, nullptr)};

    return s.Value();
}

template <typename T, typename S>
typename TypedArray<T, S>::CType TypedArray<T, S>::Min() const {
    PyScalar<CType> s{PyArray_Min(data.values.PyArrayObj(), NPY_MAXDIMS, nullptr)};

    return s.Value();
}

template <typename T, typename S>
typename TypedArray<T, S>::CType TypedArray<T, S>::Max() const {
    PyScalar<CType> s{PyArray_Max(data.values.PyArrayObj(), NPY_MAXDIMS, nullptr)};

    return s.Value();
}

template <typename T, typename S>
TypedArray<T, S> TypedArray<T, S>::FromFile(const std::string &path, FileLoadOptions options) {
    FILE* fp = fopen(path.c_str(), "rb");

    if (fp == nullptr)
        throw std::runtime_error("File could not be opened");

    npy_intp dims[] = {0};

    PyRef dummyArray;

    PyObject* arr = PyArray_SimpleNew(1, dims, NPY_INT);

    PyArray_Descr* desc = PyArray_DESCR(reinterpret_cast<PyArrayObject*>(arr));

    char sep[24] = " ";

    std::strncpy(sep, options.sep.c_str(), 24);

    TypedArray<T, S> result{0}; //PyArray_FromFile(fp, desc, -1, sep)};

    fclose(fp);

    return result;
}

template <typename T>
npy_intp sparse_size(const TypedArray<T, array_storage::SparseCRS<T>>& a) {
    return 0;// PyArray_Size(a.indices.PyObj());
}

template <typename T>
T Dot_(const TypedArray<T, array_storage::Dense<T>>& a, const TypedArray<T, array_storage::SparseCRS<T>>& b)
{
    T result{0};

    T const * valA = a.Values().Data();
    T const * valB = b.Values().Data();

    auto const * ind = b.data.indices.Data();

    for (npy_intp i = 0; i < sparse_size(b); ++i)
        result += valB[i] * valA[ind[i]];

    return result;
}

template <typename T>
T Dot_(const TypedArray<T, array_storage::SparseCRS<T>>& a, const TypedArray<T, array_storage::Dense<T>>& b)
{
    return Dot_<T>(b, a);
}

template <typename T>
T Dot_(const TypedArray<T, array_storage::Dense<T>>& a, const TypedArray<T, array_storage::Dense<T>>& b)
{
    PyScalar<typename detail::NpTypeTraits<T>::CType> s{PyArray_InnerProduct(a.Values().PyObj(), b.Values().PyObj())};

    return s.Value();
}

template <typename T, typename S>
template <typename OtherStorage>
typename TypedArray<T, S>::CType TypedArray<T, S>::Dot(const TypedArray<T, OtherStorage> &other)
{
    if (data.values.Dimensions() != other.Dimensions())
        throw std::runtime_error("Wrong size");

    return Dot_(*this, other);
}

template <typename T, typename S>
PyArray<T>& TypedArray<T, S>::Values()
{
    return data.values;
};

template <typename T, typename S>
const PyArray<T>& TypedArray<T, S>::Values() const
{
    return data.values;
}

template <typename T, typename S>
DimensionsType TypedArray<T, S>::Dimensions() const {
    return Values().Dimensions();
}

template <typename T, typename S>
T& TypedArray<T, S>::operator()(npy_intp idx1) {
    return data.values(idx1);
};

template <typename T, typename S>
T& TypedArray<T, S>::operator()(npy_intp idx1, npy_intp idx2) {
    return data.values(idx1, idx2);
};

template <typename T, typename S>
T& TypedArray<T, S>::operator()(npy_intp idx1, npy_intp idx2, npy_intp idx3) {
    return data.values(idx1, idx2, idx3);
};

template <typename T, typename S>
TypedArray<T, S>& TypedArray<T, S>::operator=(TypedArray<T, S>&& other)
{
    data = std::move(other.data);

    return *this;
};

template <typename T, typename S>
TypedArray<T, S>& TypedArray<T, S>::operator=(const TypedArray<T, S>& other)
{
    data = S{other.data};

    return *this;
};

template <typename T, typename S>
TypedArray<T, S>::TypedArray(TypedArray<T, S>&& other)
    : data(std::move(other.data))
{
};

template <typename T, typename S>
TypedArray<T, S>::TypedArray(const TypedArray<T, S>& other)
    : data(other.data)
{

};

} // xdata
