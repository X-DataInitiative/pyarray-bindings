#pragma once

#include <iostream>

#include <tick/array/Fwd.hpp>
#include <tick/array/TypeTraits.hpp>
#include <tick/array/PyRef.hpp>
#include <tick/array/PyNdArray.hpp>

namespace tick {

template <typename T>
class PyScalar : public NdArray<T, 0> {
public:
    using Base = NdArray<T, 0>;
    using Traits = typename detail::NpTypeTraits<T>;
    using CType = typename Traits::CType;

    PyScalar();
    explicit PyScalar(PyObject*&& pyObj);
    explicit PyScalar(const CType& val);

    ~PyScalar();

    CType& operator()();
    CType& value();
};

}  // namespace tick

#include <tick/array/PyScalar.tpp>
