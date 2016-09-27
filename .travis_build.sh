#!/bin/sh

mkdir build && cd build

cmake .. -DPYTHON_NUMPY_INCLUDE_DIR=/usr/lib/python3/dist-packages/numpy/core/include \
	 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.4m.so \
	 -DPYTHON_INCLUDE_DIR=/usr/include/python3.4m \
 	&& make && ./test/xdata_array_test
