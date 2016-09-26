#define NO_IMPORT_ARRAY

#include <xdata/array/TypedArray.hpp>

int somecrazysymbol_b;

template class xdata::TypedArray<double, xdata::array_storage::Dense<double>>;
template class xdata::TypedArray<double, xdata::array_storage::SparseCRS<double>>;
//template class xdata::TypedArray<long, xdata::array_storage::Dense>;
//template class xdata::TypedArray<long, xdata::array_storage::SparseCRS>;
