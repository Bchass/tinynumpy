[![Python package](https://github.com/Bchass/tinynumpy/actions/workflows/python-package.yml/badge.svg)](https://github.com/Bchass/tinynumpy/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/Bchass/tinynumpy/graph/badge.svg?token=fpx7bXEqTx)](https://codecov.io/gh/Bchass/tinynumpy)

tinynumpy
=========

A lightweight, pure Python, numpy compliant ndarray class.

This module is intended to allow libraries that depend on numpy, but
do not make much use of array processing, to make numpy an optional
dependency. This might make such libaries better available, also on
platforms like Pypy and Jython.


Features
--------

* The ndarray class has all the same properties as the numpy ndarray
  class.
* Pretty good compliance with numpy in terms of behavior (such as views).
* Can be converted to a numpy array (with shared memory).
* Can get views of real numpy arrays (with shared memory).
* Support for wrapping ctypes arrays, or provide ctypes pointer to data.
* Pretty fast for being pure Python.
* Works on Python 3.x, Pypy and Jython.

Caveats
-------

* ndarray.flat iterator cannot be indexed (it is a generator).
* Support for data types limited to bool, uin8, uint16, uint32, uint64,
  int8, int16, int32, int64, float32, float64.
* Functions that calculate statistics on the data are much slower, since
  the iteration takes place in Python.
* Assigning via slicing is usually pretty fast, but can be slow if the
  striding is unfortunate.


Examples
--------

```python3

>>> from tinynumpy import tinynumpy as tnp

>>> a = tnp.array([[1, 2, 3, 4],[5, 6, 7, 8]])

>>> a
array([[ 1.,  2.,  3.,  4.],
    [ 5.,  6.,  7.,  8.]], dtype='float64')

>>> a[:, 2:]
array([[ 3.,  4.],
    [ 7.,  8.]], dtype='float64')

>>> a[:, ::2]
array([[ 1.,  3.],
    [ 5.,  7.]], dtype='float64')

>>> a.shape
(2, 4)

>>> a.shape = 4, 2

>>> a
array([[ 1.,  2.],
    [ 3.,  4.],
    [ 5.,  6.],
    [ 7.,  8.]], dtype='float64')

>>> b = a.ravel()

>>> a[0, 0] = 100

>>> b
array([ 100.,  2.,  3.,  4.,  5.,  6.,  7.,  8.], dtype='float64')
```

Build locally
--------

> [!NOTE]  
> Aiming to relaunch on PyPi once [v1.2.1](https://github.com/Bchass/tinynumpy/milestone/1) is completed.

```
>>> pip install build
>>> python -m build
>>> pip install dist/tinynumpy-1.2.1.tar.gz
```