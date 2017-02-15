#include <iostream>

#include <cstring>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <tick/array/PyRef.hpp>

namespace tick {

#if PY_VERSION_HEX >= 0x03000000
int InitNumpy_() { import_array(); return 0; }
#else
void InitNumpy_() { import_array(); }
#endif

PyRef::PyRef()
    : pyObj(nullptr)
{

}

//PyRef::PyRef(PyObject** pyObj)
//    : pyObj(*pyObj)
//{
//    *pyObj = nullptr;
//}

PyRef::PyRef(PyObject*&& obj)
    : pyObj(obj)
{
    if (pyObj == nullptr)
        throw std::runtime_error("PyRef initialized with null-pointer!");

    obj = nullptr;
}

PyRef::PyRef(const PyRef &other)
    : pyObj(other.pyObj) {
    Py_XINCREF(pyObj);
}

PyRef::PyRef(PyRef &&other)
    : pyObj(other.pyObj)
{
    other.pyObj = nullptr;
}

PyRef& PyRef::operator=(const PyRef &other)
{
    this->pyObj = other.pyObj;

    Py_XINCREF(pyObj);

    return *this;
}

PyRef& PyRef::operator=(PyRef &&other) {
    this->pyObj = other.pyObj;

    other.pyObj = nullptr;

    return *this;
}

PyRef::~PyRef() {
    Py_XDECREF(pyObj);
}

bool PyRef::operator==(const PyRef &rhs) const
{
    return pyObj == rhs.pyObj;
}

bool PyRef::operator!=(const PyRef &rhs) const
{
    return !(rhs == *this);
}

void PyRef::setPyObj(PyObject *pyObj)
{
    this->pyObj = pyObj;

    Py_XINCREF(pyObj);
}

void PyRef::Init() {
    if (PyArray_API == nullptr) {
        InitNumpy_();
    }
}

std::size_t PyRef::RefCount() const {
    if (pyObj != nullptr)
        return static_cast<std::size_t>(pyObj->ob_refcnt);
    else
        return std::size_t{};
}

PyRef PyRef::Take(PyObject *&obj) {
    Py_XINCREF(obj);

    PyRef ref{std::move(obj)};

    obj = nullptr;

    return ref;
}


}
