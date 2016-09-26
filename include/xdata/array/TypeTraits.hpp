#pragma once

#include <type_traits>
#include <numpy/ndarraytypes.h>

namespace xdata {
namespace detail {

template <typename T>
struct NpTypeTraitsBase
{
    using CType = T;
};

template <typename T>
struct NpTypeTraits
{
    static const NPY_TYPES NpType = NPY_VOID;

    static_assert(!std::is_same<T, T>{}, "No specialization for type T");
};

template <>
struct NpTypeTraits<double> : NpTypeTraitsBase<double>
{
    static const NPY_TYPES NpType = NPY_DOUBLE;
};

template <>
struct NpTypeTraits<long> : NpTypeTraitsBase<long>
{
    static const NPY_TYPES NpType = NPY_LONG;
};

template <>
struct NpTypeTraits<unsigned long> : NpTypeTraitsBase<unsigned long>
{
    static const NPY_TYPES NpType = NPY_ULONG;
};

template <>
struct NpTypeTraits<bool> : NpTypeTraitsBase<bool>
{
    static const NPY_TYPES NpType = NPY_BOOL;
};

template <> struct NpTypeTraits<float>          : public NpTypeTraits<double> {};
template <> struct NpTypeTraits<short>          : public NpTypeTraits<long> {};
template <> struct NpTypeTraits<int>            : public NpTypeTraits<long> {};
template <> struct NpTypeTraits<unsigned short> : public NpTypeTraits<unsigned long> {};
template <> struct NpTypeTraits<unsigned int>   : public NpTypeTraits<unsigned long> {};

}
}
