#pragma once

#include <array>
#include <iostream>

#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>

#include <tick/array/TypeTraits.hpp>
#include <tick/array/PyRef.hpp>

namespace tick {

template <typename T, unsigned Ndim>
class NdArray {
public:
    using Traits = typename detail::NpTypeTraits<T>;
    using CType = typename Traits::CType;

    //std::array<std::size_t, Ndim> dimensions;

    explicit NdArray(PyRef&& other)
        : value(std::move(other)) {

        std::cout << "Active refcnt: " << value.RefCount() << std::endl;
    }

    explicit NdArray(std::array<std::size_t, Ndim> dims)
        : value(PyArray_SimpleNew(dims.size(), reinterpret_cast<npy_intp*>(dims.data()), Traits::NpType)) {

    }

    NdArray(const NdArray& other)
        : value(PyArray_SimpleNew(other.GetNDimensions(), reinterpret_cast<npy_intp*>(other.Dimensions().data()), Traits::NpType)) {

    }

    NdArray(NdArray&& other)
        : value(std::move(other.value)) {

    }

    NdArray& operator=(const NdArray& other) {
        value = PyArray_SimpleNew(other.GetNDimensions(), reinterpret_cast<npy_intp*>(other.Dimensions().data()), Traits::NpType);

        return *this;
    }

    NdArray& operator=(NdArray&& other) {
        value = std::move(other.value);

        return *this;
    }

    T* Data() {
        return reinterpret_cast<CType *>(PyArray_DATA(PyArrayObj()));
    }

    T& at(std::size_t i) {
        return *(Data() + i);
    }

    const T& at(std::size_t i) const {
        return *(Data() + i);
    }

    T& operator[](std::size_t i) {
        return at(i);
    }

    const T& operator[](std::size_t i) const {
        return at(i);
    }

    T& Begin() {
        return at(0);
    }

    const T& Begin() const {
        return at(0);
    }

    T& End() {
        return at(Size());
    }

    const T& End() const {
        return at(Size());
    }

    std::size_t Size() const {
        return PyArray_SIZE(PyArrayObj());
    }

    std::array<std::size_t, Ndim> Dimensions() const {
        std::array<std::size_t, Ndim> res;
        npy_intp* dims = PyArray_DIMS(PyArrayObj());

        for (std::size_t i = 0; i < res.size(); ++i) res[i] = dims[i];

        return res;
    };

    constexpr unsigned GetNDimensions() const { return Ndim; }

    PyRef& GetPyRef() {
        return value;
    }

    const PyRef& GetPyRef() const {
        return value;
    }

    PyArrayObject* PyArrayObj() const {
        return reinterpret_cast<PyArrayObject *>(GetPyRef().PyObj());
    }

    void Fill(CType value) {
        std::fill(&Begin(), &End(), value);
    }

    void Zero(CType value) {
        std::fill(&Begin(), &End(), 0);
    }

    // Math functions
    CType Sum() const {
        PyRef sum{PyArray_Sum(PyArrayObj(), NPY_MAXDIMS, Traits::NpType, nullptr)};

        CType result;

        PyArray_ScalarAsCtype(sum.PyObj(), &result);

        return result;
    }

    double Mean() const {
        PyRef mean{PyArray_Mean(PyArrayObj(), NPY_MAXDIMS, Traits::NpType, nullptr)};

        double result;

        PyArray_ScalarAsCtype(mean.PyObj(), &result);

        return result;
    }

private:
    PyRef value;
};



//template <typename T>
//class Array2D : public Ndarray<T, 2> {
//public:
//    using Base = Ndarray<T, 2>;
//
//    Array2D(std::size_t size_x, std::size_t size_y) : Base({size_x, size_y}) {}
//
//    T& operator()(std::size_t i, std::size_t j) { return this->pyArray.at(i * this->dimensions[0] + j); }
//};
//
//using ScalarDouble = Scalar<double>;
//using ArrayDouble = Array<double>;
//using ArrayDouble2D = Array2D<double>;

}