#pragma once

#include <xdata/array/PyScalar.hpp>

namespace xdata {

namespace detail {

template <typename T>
struct PyObjectFromValue {
    PyObject* operator()(const T& value) { return nullptr; }
};

template <>
struct PyObjectFromValue<double> {
    PyObject* operator()(const double& value) { return PyFloat_FromDouble(value); }
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

}

template <typename T>
PyScalar<T>::PyScalar(PyObject *&&pyObj)
    : value(std::move(pyObj))
{
    if (!PyArray_CheckAnyScalar(PyObj()))
        throw std::runtime_error("Object is not a scalar type!");
}

template <typename T>
PyScalar<T>::PyScalar(const CType & val)
    : value(detail::PyObjectFromValue<CType>{}(val))
{
}

template <typename T>
PyScalar<T>::~PyScalar()
{
}

template <typename T>
typename PyScalar<T>::CType PyScalar<T>::Value()
{
    if (PyArray_Check(PyObj())) {
        return (reinterpret_cast<CType*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(PyObj()))))[0];
    } else {
        CType out{};

        PyArray_ScalarAsCtype(PyObj(), &out);

        return out;
    }
}

template <typename T>
PyObject* PyScalar<T>::PyObj() const
{
    return value.PyObj();
}

} // xdata