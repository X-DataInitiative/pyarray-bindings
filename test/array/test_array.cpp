#include <fstream>
#include <gtest/gtest.h>

#include <numpy/npy_3kcompat.h>

#include <tick/array/PyRef.hpp>
#include <tick/array/PyNdArray.hpp>
#include <tick/array/PyScalar.hpp>

TEST(PyRef, Constructor) {
    EXPECT_ANY_THROW(tick::PyRef{[]{ return nullptr; }()});

    tick::PyRef r = tick::PyRef(PyLong_FromLong(0));

    EXPECT_GT(r.RefCount(), 0);
}

TEST(PyRef, RefCount) {
    tick::PyRef l{PyLong_FromLong(1337)};

    EXPECT_EQ(1, l.RefCount());

    {
        tick::PyRef r = l;

        EXPECT_EQ(2, l.RefCount());
        EXPECT_EQ(2, r.RefCount());
    }

    {
        tick::PyRef r = std::move(l);

        EXPECT_EQ(0, l.RefCount());
        EXPECT_EQ(1, r.RefCount());
        EXPECT_EQ(nullptr, l.PyObj());
    }

    EXPECT_EQ(0, l.RefCount());
}

TEST(PyScalar, First) {
    const double val = 94.51;
    tick::PyScalar<double> scalar(val);

    EXPECT_EQ(scalar.GetNDimensions(), 0);
    EXPECT_EQ(scalar(), val);
    EXPECT_EQ(scalar.value(), val);
    EXPECT_EQ(scalar.at(0), val);
    EXPECT_EQ(scalar[0], val);

    const double other_val = 23.89;
    scalar() = other_val;

    EXPECT_EQ(scalar(), other_val);
}

TEST(PyScalar, Sum) {
    const double val = 31.69;
    tick::PyScalar<double> scalar(val);

    EXPECT_EQ(scalar.Sum(), val);
}

TEST(PyScalar, Mean) {
    const double val = 87.71;
    tick::PyScalar<double> scalar(val);

    EXPECT_DOUBLE_EQ(scalar.Mean(), val);
}

TEST(PyArray, First) {
    tick::Array<double> arr(100);

    arr.at(50) = 42.42;

    EXPECT_DOUBLE_EQ(arr.at(50), 42.42);
}

TEST(PyArray, Fill) {
    tick::Array<double> arr(100);

    arr.Fill(1234.5678);

    EXPECT_DOUBLE_EQ(arr.Mean(), 1234.5678);
    EXPECT_DOUBLE_EQ(arr.Sum(), 1234.5678 * 100);
}

TEST(PyArray, Zero) {
    tick::Array<double> arr(100);

    arr.Fill(1234.5678);

    EXPECT_DOUBLE_EQ(arr.Mean(), 1234.5678);
    EXPECT_DOUBLE_EQ(arr.Sum(), 1234.5678 * 100);
}


//TEST(PyArray, RefConstructor) {
//    EXPECT_ANY_THROW(xdata::PyArray<int>{nullptr});
//    EXPECT_ANY_THROW(xdata::PyArray<int>{PyLong_FromLong(1337)});
//
//    npy_intp dims[1] = {0};
//    EXPECT_NO_THROW(xdata::PyArray<int>{PyArray_SimpleNew(1, dims, NPY_LONG)});
//}
//
//TEST(NpArrayTest, Sizes) {
//    auto arr = xdata::TypedArray<double>{123, 234};
//
//    const auto dims = arr.Dimensions();
//
//    EXPECT_EQ(3, dims.size());
//    EXPECT_EQ(123, dims[0]);
//    EXPECT_EQ(234, dims[1]);
//
//}

//TEST(NpArrayTest, FromFile) {
//    const std::string filename = "testarray.mat";
//
//    {
//        std::ofstream of{filename};
//
//        of << "1 2 3 4 -4\n -3 -2 -1 -0";
//    }
//
//    auto arr = xdata::TypedArray<double>::FromFile(filename);
//
//    const auto dims = arr.Dimensions();
//
//    EXPECT_EQ(3, dims.size());
//    EXPECT_EQ(9, dims[0]);
//    EXPECT_EQ(1, dims[1]);
//    EXPECT_EQ(1, dims[2]);
//
//    {
//        remove(filename.c_str());
//    }
//}

//TEST(NpArrayTest, Transpose) {
//    auto arr = xdata::TypedArray<long>{100, 50};
//
//    arr.Transposed();
//
//    {
//        const auto dims = arr.Dimensions();
//
//        EXPECT_EQ(100, dims[0]);
//        EXPECT_EQ(50, dims[1]);
//    }
//
//    auto transposedArr = arr.Transposed();
//
//    {
//        const auto dims = transposedArr.Dimensions();
//
//        EXPECT_EQ(50, dims[0]);
//        EXPECT_EQ(100, dims[1]);
//    }
//}
//
//TEST(NpArrayTest, Flattened) {
//    auto arr = xdata::TypedArray<long>{100, 50};
//
//    {
//        const auto dims = arr.Dimensions();
//
//        EXPECT_EQ(100, dims[0]);
//        EXPECT_EQ(50, dims[1]);
//    }
//
//    auto flatArr = arr.Flattened();
//
//    {
//        const auto dims = flatArr.Dimensions();
//
//        EXPECT_EQ(100 * 50, dims[0]);
//    }
//
//    EXPECT_NE(arr.Data(), flatArr.Data());
//}
//
//TEST(NpArrayTest, Flatten) {
//    auto arr = xdata::TypedArray<long>{100, 50};
//    auto flatArr = arr.FlattenedNoCopy();
//
//    {
//        const auto dims = arr.Dimensions();
//
//        EXPECT_EQ(100, dims[0]);
//        EXPECT_EQ(50, dims[1]);
//    }
//
//    {
//        const auto dims = flatArr.Dimensions();
//
//        EXPECT_EQ(100 * 50, dims[0]);
//    }
//
//    EXPECT_EQ(arr.Data(), flatArr.Data());
//}

//TEST(TypedArrayTest, Copy) {
//    xdata::TypedArray<double> arr(100);
//
//    // should be copy
//    xdata::TypedArray<double> arr2 = arr;
//
//    EXPECT_NE(arr.Values().Data(), arr2.Values().Data());
//}
//
//TEST(TypedArrayTest, Move) {
//    xdata::TypedArray<double> arr(100);
//
//    double* olddataloc = arr.Values().Data();
//
//    xdata::TypedArray<double> arr2 = std::move(arr);
//
//    EXPECT_EQ(nullptr, arr.Values().PyObj());
//    EXPECT_EQ(olddataloc, arr2.Values().Data());
//}
//
//TEST(TypedArrayTest, Fill) {
//    xdata::TypedArray<double> arr(100);
//
//    arr.Fill(1337);
//
//    EXPECT_EQ(1337, arr.Values().Data()[50]);
//}
//
//TEST(TypedArrayTest, Sum) {
//    xdata::TypedArray<double> arr(123, 234);
//
//    arr.Fill(2);
//
//    EXPECT_EQ(123 * 234 * 2, arr.Sum());
//}
//
//TEST(TypedArrayTest, Mean) {
//    xdata::TypedArray<double> arr(123, 234);
//
//    arr.Fill(2);
//
//    EXPECT_EQ(2.0, arr.Mean());
//
//    std::vector<long> vals = {47, 53, 80, 9, 33, 25, 11, 13, 52, 1, 78, 50, 54, 44, 64, 89, 32, 6, 99, 73, 75, 96, 57, 30, 61};
//
//    xdata::TypedArray<long> arr2(vals.size());
//
//    std::copy(std::begin(vals), std::end(vals), arr2.Values().Data());
//
//    EXPECT_EQ(1232, arr2.Sum());
//    EXPECT_EQ(49.28, arr2.Mean());
//}
//
//TEST(TypedArrayTest, Dot) {
//    xdata::TypedArray<long> arr1(11);
//    xdata::TypedArray<long> arr2(11);
//
//    arr1.Fill(1);
//    arr2.Fill(2);
//
//    ASSERT_EQ(11 * 1, arr1.Sum());
//    ASSERT_EQ(11 * 2, arr2.Sum());
//
//    EXPECT_EQ(22, arr1.Dot(arr2));
//}
//
//TEST(TypedArrayTest, Generate) {
//    xdata::TypedArray<long> arr1(10, 20, 30);
//
//    int i = 0;
//    arr1.Generate([&i](){ return i++; });
//
//    EXPECT_EQ(17997000, arr1.Sum());
//}
//
//TEST(TypedArrayTest, SparseDot) {
//    xdata::ArrayDouble arrayDouble{10, 10};
//    xdata::ArrayDoubleSparse arrayDoubleSparse{10, 10};
//
//   // arrayDoubleSparse(0, 0) = 2;
//
//    xdata::ArrayLong arrayLong{10, 10};
//    xdata::ArrayLongSparse arrayLongSparse{10, 10};
//
//    EXPECT_NE(0, arrayDouble.Dot(arrayDoubleSparse));
//}
//
//TEST(TypedArrayTest, DenseIndexing) {
//    xdata::ArrayDouble arrayDouble{10, 10, 10};
//
//    arrayDouble(0, 5) = 5.0;
//    EXPECT_EQ(arrayDouble(0, 5), 5.0);
//
//    EXPECT_EQ(arrayDouble.Values().Data(), &arrayDouble(0));
//    EXPECT_EQ(arrayDouble.Values().Data() + 5 + (10), &arrayDouble(0, 1, 5));
//    EXPECT_EQ(arrayDouble.Values().Data() + 5 + (10) + (10 * 10), &arrayDouble(1, 1, 5));
//    EXPECT_EQ(arrayDouble.Values().Data() + 5, &arrayDouble(0, 0, 5));
//
//    EXPECT_EQ(&arrayDouble(0, 5), &arrayDouble(0, 5, 0));
//    EXPECT_EQ(&arrayDouble(0, 11, 0), &arrayDouble(1, 1, 0));
//    EXPECT_EQ(&arrayDouble(1, -1, 5), &arrayDouble(0, 9, 5));
//}
//
//template <typename T>
//class PTypedArrayTest : public ::testing::Test {
//
//};
//
//TYPED_TEST_CASE_P(PTypedArrayTest);
//
//#include <numeric>
//
//TYPED_TEST_P(PTypedArrayTest, Indexing) {
//    xdata::TypedArray<TypeParam> arr(10, 11, 12);
//
//    arr.Fill(TypeParam{1337});
//
//    auto dims = arr.Dimensions();
//
//    EXPECT_EQ(10, dims[0]);
//    EXPECT_EQ(11, dims[1]);
//    EXPECT_EQ(12, dims[2]);
//
//    TypeParam x = std::accumulate(arr.Values().Data(), arr.Values().Data() + (10 * 11 * 12), 0);
//    ASSERT_EQ(TypeParam{10 * 11 * 12 * 1337}, x);
//
//    EXPECT_EQ(TypeParam{10 * 11 * 12 * 1337}, arr.Sum());
//}
//
//REGISTER_TYPED_TEST_CASE_P(PTypedArrayTest,
//                           Indexing);

//typedef ::testing::Types<int, unsigned int, long, unsigned long, float, double> TypedArrayTestTypes;
//INSTANTIATE_TYPED_TEST_CASE_P(My, PTypedArrayTest, TypedArrayTestTypes);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    Py_SetProgramName(reinterpret_cast<wchar_t*>(argv[0]));
    Py_Initialize();

    std::cout << "Python version: " << Py_GetVersion() << '\n';

    tick::PyRef::Init();

    const int r = RUN_ALL_TESTS();

    Py_Finalize();

    return r;
}