#pragma once

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <tick/array/PyScalar.hpp>

namespace tick {

namespace detail {

template <typename T>
struct PyObjectFromValue {
    PyObject* operator()(const T& value) { return nullptr; }
};

template <>
struct PyObjectFromValue<double> {
    PyObject* operator()(const double& value) {
        PyObject* const ptr = PyArray_SimpleNew(0, nullptr, NpTypeTraits<double>::NpType);

        *reinterpret_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ptr))) = value;

        return ptr;
    }
};

template <>
struct PyObjectFromValue<long> {
    PyObject* operator()(const long& value) { return PyLong_FromLong(value); }
};

template <>
struct PyObjectFromValue<unsigned long> {
    PyObject* operator()(const unsigned long& value) { return PyLong_FromUnsignedLong(value); }
};

template <>
struct PyObjectFromValue<bool> {
    PyObject* operator()(const bool& value) { return PyLong_FromLong(value);; }
};

}  // namespace detail

template <typename T>
PyScalar<T>::PyScalar()
    : Base(std::array<std::size_t, 0>{}) {
}

template <typename T>
PyScalar<T>::PyScalar(PyRef&& other)
    : Base(std::move(other)) {
    if (!PyArray_CheckScalar(Base::PyArrayObj()))
        throw std::runtime_error("Object is not a scalar type!");
};

template <typename T>
PyScalar<T>::PyScalar(const CType & val)
    : Base(std::array<std::size_t, 0>{}) {
    value() = val;
}

template <typename T>
PyScalar<T>::~PyScalar()
{}

template <typename T>
typename PyScalar<T>::CType& PyScalar<T>::value() {
    return Base::at(0);
}

template <typename T>
typename PyScalar<T>::CType& PyScalar<T>::operator()() {
    return value();
}



}  // namespace xdata