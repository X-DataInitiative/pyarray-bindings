#pragma once

#include <iostream>

#include <xdata/array/Fwd.hpp>
#include <xdata/array/TypeTraits.hpp>
#include <xdata/array/PyRef.hpp>

namespace xdata {

template <typename T>
class PyScalar {
public:
    using Traits = typename detail::NpTypeTraits<T>;
    using CType = typename Traits::CType;

    explicit PyScalar(PyObject*&& pyObj);
    explicit PyScalar(const CType& val);

    ~PyScalar();

    CType Value();

    PyObject* PyObj() const;
private:
    PyRef value;
};

}

#include <xdata/array/PyScalar.tpp>

