#pragma once

#include <xdata/array/PyArray.hpp>

#include <numpy/arrayobject.h>

namespace xdata {

template <typename T>
PyArray<T>::PyArray() {

}

template <typename T>
PyArray<T>::PyArray(PyRef &&ref)
    : data{std::move(ref)}
{
    std::cout << "PyArray ref constructor" << std::endl;

    if (!PyArray_Check(data.PyObj()))
        throw std::runtime_error("Object is not a Numpy array!");

    if (!IsWellFormed())
        throw std::runtime_error("Data is not well-formed (C-style contiguous)");

    if (PyArray_TYPE(PyArrayObj()) != Traits::NpType)
        throw std::runtime_error("Types are wrong");
}

template <typename T>
PyArray<T>::PyArray(std::array<npy_intp, 1> dims)
    : data{PyArray_SimpleNew(1, dims.data(), Traits::NpType)}
{
}

template <typename T>
PyArray<T>::PyArray(std::array<npy_intp, 2> dims)
    : data{PyArray_SimpleNew(2, dims.data(), Traits::NpType)}
{
}

template <typename T>
PyArray<T>::PyArray(std::array<npy_intp, 3> dims)
    : data{PyArray_SimpleNew(3, dims.data(), Traits::NpType)}
{
}

template <typename T>
PyObject* PyArray<T>::PyObj() const
{
    return data.PyObj();
}

template <typename T>
PyArrayObject* PyArray<T>::PyArrayObj() const
{
    return reinterpret_cast<PyArrayObject*>(PyObj());
}

template <typename T>
typename PyArray<T>::CType* PyArray<T>::Data() const
{
    return reinterpret_cast<PyArray<T>::CType*>(PyArray_DATA(PyArrayObj()));
}

template <typename T>
DimensionsType PyArray<T>::Dimensions() const {
    DimensionsType result{};

    if (data.PyObj() != nullptr) {
        const std::size_t ndim = static_cast<std::size_t>(NDims());
        const npy_intp* dims = PyArray_DIMS(PyArrayObj());

        for (int i = 0; i < result.size(); ++i) result[i] = i < ndim ? dims[i] : 0;
    }

    return result;
}

template <typename T>
int PyArray<T>::NDims() const {
    return PyArray_NDIM(PyArrayObj());
}

template <typename T>
bool PyArray<T>::IsWellFormed() const {
    return PyArray_ISCARRAY_RO(PyArrayObj());
}

template <typename T>
typename PyArray<T>::CType& PyArray<T>::operator()(npy_intp idx1)
{
    return *(reinterpret_cast<T*>(PyArray_GETPTR1(PyArrayObj(), idx1)));
}

template <typename T>
typename PyArray<T>::CType& PyArray<T>::operator()(npy_intp idx1, npy_intp idx2)
{
    return *(reinterpret_cast<T*>(PyArray_GETPTR2(PyArrayObj(), idx1, idx2)));
}

template <typename T>
typename PyArray<T>::CType& PyArray<T>::operator()(npy_intp idx1, npy_intp idx2, npy_intp idx3)
{
    return *(reinterpret_cast<CType*>(PyArray_GETPTR3(PyArrayObj(), idx1, idx2, idx3)));
}

template <typename T>
const typename PyArray<T>::CType& PyArray<T>::operator()(npy_intp idx1) const
{
    return *(reinterpret_cast<CType*>(PyArray_GETPTR1(PyArrayObj(), idx1)));
}

template <typename T>
const typename PyArray<T>::CType& PyArray<T>::operator()(npy_intp idx1, npy_intp idx2) const
{
    return *(reinterpret_cast<CType*>(PyArray_GETPTR2(PyArrayObj(), idx1, idx2)));
}

template <typename T>
const typename PyArray<T>::CType& PyArray<T>::operator()(npy_intp idx1, npy_intp idx2, npy_intp idx3) const
{
    return *(reinterpret_cast<CType*>(PyArray_GETPTR3(PyArrayObj(), idx1, idx2, idx3)));
}

template <typename T>
PyArray<T>::PyArray(const PyArray<T> &other)
    : data()
{
    PyObject* ptr = PyArray_FROM_OTF(other.PyObj(), Traits::NpType, NPY_ARRAY_ENSURECOPY);

    std::cout << "py...: " << (void*) other.PyObj() << std::endl;
    std::cout << "...: " << (void*) ptr << std::endl;

    auto dims = other.Dimensions();
    std::cout << "Copying array: " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;

    if (ptr != nullptr) {
        std::cout << "Result of copy was not null" << std::endl;

        data = PyRef{std::move(ptr)};

    } else {
        std::cout << "Result of copy was null" << std::endl;

        data = PyRef{PyArray_SimpleNew(1, other.Dimensions().data(), Traits::NpType)};
    }

    std::cout << "OK" << std::endl;

}

template <typename T>
PyArray<T>::PyArray(PyArray<T>&& other)
    : data(std::move(other.data))
{
}

template <typename T>
PyArray<T>& PyArray<T>::operator=(const PyArray<T>& other)
{
    data = PyRef{PyArray_FROM_OTF(other.PyObj(), Traits::NpType, NPY_ARRAY_ENSURECOPY)};

    return *this;
}

template <typename T>
PyArray<T>& PyArray<T>::operator=(PyArray<T>&& other)
{
    data = std::move(other.data);

    std::cout << "Move assignment, pyarray" << std::endl;

    return *this;
}


//template <typename T>
//PyArray<T> PyArray<T>::Transposed() const
//{
//    return PyArray{PyArray_Transpose(PyArrayObj(), nullptr)};
//}
//
//template <typename T>
//PyArray<T> PyArray<T>::Flattened() const
//{
//    return PyArray{PyArray_Flatten(PyArrayObj(), NPY_ANYORDER)};
//}
//
//template <typename T>
//PyArray<T> PyArray<T>::FlattenedNoCopy() const
//{
//    return PyArray{PyArray_Ravel(PyArrayObj(), NPY_CORDER)};
//}

}
