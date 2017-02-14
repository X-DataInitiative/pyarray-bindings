#pragma once

#include <cstddef>
#include <string>
#include <array>

#include <tick/array/Fwd.hpp>

namespace tick {

class PyRef {

public:
    PyRef();

    explicit PyRef(PyObject*&& obj);

    PyRef(const PyRef& other);
    PyRef(PyRef&& other);

    PyRef& operator=(const PyRef& other);
    PyRef& operator=(PyRef&& other);

    virtual ~PyRef();

    bool operator==(const PyRef &rhs) const;
    bool operator!=(const PyRef &rhs) const;

    inline PyObject *PyObj() const { return pyObj; }

    void setPyObj(PyObject *pyObj);

    std::size_t RefCount() const;

    static void Init();

private:
    PyObject* pyObj;
};

}
