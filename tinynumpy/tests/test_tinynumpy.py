# -*- coding: utf-8 -*-
# Copyright (c) 2014, Almar Klein and Wade Brainerd
# tinynumpy is distributed under the terms of the MIT License.

""" Test suite for tinynumpy
"""

import ctypes

import pytest
from pytest import raises, skip
import faulthandler

try:
    import tinynumpy.tinynumpy as tnp
except ImportError:
    import tinynumpy as tnp

# Numpy is optional. If not available, will compare against ourselves.
try:
    import numpy as np
except ImportError:
    np = tnp


def test_TESTING_WITH_NUMPY():
    # So we can see in the result whether numpy was used
    if np is None or np is tnp:
        skip('Numpy is not available')


def test_shapes_and_strides():
    
    for shape in [(9, ), (109, ), 
                  (9, 4), (109, 104), 
                  (9, 4, 5), (109, 104, 105),
                  (9, 4, 5, 6),  # not (109, 104, 105, 106) -> too big
                  ]:
        
        # Test shape and strides
        a = np.empty(shape)
        b = tnp.empty(shape)
        assert a.ndim == len(shape)
        assert a.ndim == b.ndim
        assert a.shape == b.shape
        assert a.strides == b.strides
        assert a.size == b.size
        
        # Also test repr length
        if b.size > 100:
            assert len(repr(b)) < 80
        else:
            assert len(repr(b)) > (b.size * 3)  # "x.0" for each element

def test_strides_for_shape():

    shapes_itemsize = [
        ((3,), 4, 'C', (4,)),
        ((3,), 4, 'F', (4,)),
        ((3, 4), 4, 'C', (16, 4)),
        ((3, 4), 4, 'F', (4, 12)),
        ((3, 4, 2), 4, 'C', (32, 8, 4)),
        ((3, 4, 2), 4, 'F', (4, 12, 48)), 
        ((5, 4, 3), 8, 'C', (96, 24, 8)),
        ((5, 4, 3), 8, 'F', (8, 40, 160)),
    ]

    for shape, itemsize, order, expected_strides in shapes_itemsize:

        actual_strides = tnp._strides_for_shape(shape, itemsize, order)

        dtype = f'int{itemsize * 8}' 
        a = np.empty(shape, dtype=dtype, order=order)
        numpy_strides = a.strides
        
        # check against numpy
        assert actual_strides == numpy_strides, f"For shape {shape}, order {order}: Expected {actual_strides}, got {numpy_strides}"

def test_c_order():
        a = tnp.array([1, 2, 3], order='C')
        assert a.flags['C_CONTIGUOUS'] == True
        assert a.flags['F_CONTIGUOUS'] == True

        b = tnp.array([[1, 2, 3], [4, 5, 6]], order='C')
        assert b.flags['C_CONTIGUOUS'] == True
        assert b.flags['F_CONTIGUOUS'] == False

def test_f_order():
        a = np.array([1, 2, 3], order='F')
        assert a.flags['C_CONTIGUOUS'] == True
        assert a.flags['F_CONTIGUOUS'] == True

        b = tnp.array([[1, 2, 3], [4, 5, 6]], order='F')
        assert b.flags['C_CONTIGUOUS'] == False
        assert b.flags['F_CONTIGUOUS'] == True

def test_unspecified_order():
        a = tnp.array([1, 2, 3])
        assert a.flags['C_CONTIGUOUS'] == True
        assert a.flags['F_CONTIGUOUS'] == True

        b = tnp.array([[1, 2, 3], [4, 5, 6]])
        assert b.flags['C_CONTIGUOUS'] == True
        assert b.flags['F_CONTIGUOUS'] == False

def test_empty_array():
        a = tnp.array([], order='C')
        assert a.flags['C_CONTIGUOUS'] == True
        assert a.flags['F_CONTIGUOUS'] == True

        b = tnp.array([], order='F')
        assert b.flags['C_CONTIGUOUS'] == True
        assert b.flags['F_CONTIGUOUS'] == True

def test_multiple_dimensions():
        a = tnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], order='C')
        assert a.flags['C_CONTIGUOUS'] == True
        assert a.flags['F_CONTIGUOUS'] == False

        skip()
        b = tnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], order='F')
        assert b.flags['C_CONTIGUOUS'] == False
        assert b.flags['F_CONTIGUOUS'] == True


def test_ndarray_int_conversion():
    # Test case 1: Array with size 1
    a = tnp.array([42])
    assert int(a) == 42

    # Test case 2: Array with size > 1
    b = tnp.array([1, 2, 3])
    try:
        int(b)
    except TypeError as e:
        assert str(e) == 'Only length-1 arrays can be converted to scalar'
    else:
        assert False, "Expected TypeError not raised"

    # edge scenarios
    c = tnp.array([], dtype='int32')
    try:
        int(c)
    except TypeError as e:
        assert str(e) == 'Only length-1 arrays can be converted to scalar'
    else:
        assert False, "Expected TypeError not raised"


def test_repr():
    for dtype in ['float32', 'float64', 'int32', 'int64']:
        for data in [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        ]:
            a = np.array(data, dtype)
            b = tnp.array(data, dtype)

            for l1, l2 in zip(repr(a).splitlines(), repr(b).splitlines()):
                l1, l2 = l1.rstrip(), l2.rstrip()
                l1, l2 = l1.split('dtype=')[0], l2.split('dtype=')[0]
                l1 = l1.replace(' ', '').replace('\t', '').rstrip(',)').replace('.', '')
                l2 = l2.replace(' ', '').replace('\t', '').rstrip(',)')
                assert l1 == l2

def test__float__():

    # test floating point values
    a = tnp.array([5])
    result = float(a)
    expected_result = 5.0
    assert result == expected_result

    with pytest.raises(TypeError):
        b = tnp.array([5, 3])
        result = float(b)


def test__add__():

    # test classic addition
    a = tnp.array([1, 2, 3])
    result = a + 1
    expected_result = tnp.array([2, 3, 4], dtype='int64')
    assert all(result == expected_result)

    a2 = tnp.array([1, 2, 3])
    b2 = tnp.array([4, 5, 6])
    result2 = a2 + b2
    expected_result2 = tnp.array([5, 7, 9], dtype='int64')   
    assert all(result2 == expected_result2)

    a4 = tnp.array([5,10,15])
    result = a4 + 1.5
    expected_result = tnp.array([6.5, 11.5, 16.5], dtype='float64')
    assert all(result == expected_result)

    # test __radd__
    c = tnp.array([1, 2, 3])
    result = 1 + c
    expected_result = tnp.array([2, 3, 4], dtype='int64')
    assert all(result == expected_result)

    a3 = tnp.array([1, 2, 3])
    b3 = tnp.array([4, 5, 6])
    result3 = b3 + a3
    expected_result3 = tnp.array([5, 7, 9], dtype='int64')   
    assert all(result3 == expected_result3)


def test__sub__():

    # test classic subtraction
    a = tnp.array([1, 2, 3])
    result = a - 1
    expected_result = tnp.array([0, 1, 2], dtype='int64')
    assert all(result == expected_result)

    b = tnp.array([5, 10, 15])
    result = b - 1.0
    expected_result = tnp.array([ 4.,  9.,  14.])
    assert all(result == expected_result)

    a2 = tnp.array([1, 2, 3])
    b2 = tnp.array([4, 5, 6])
    result2 = a2 - b2
    expected_result2 = tnp.array([-3, -3, -3], dtype='int64')   
    assert all(result2 == expected_result2)

    # test __rsub__
    c = tnp.array([1, 2, 3])
    result = 1 - c
    expected_result = tnp.array([0, 1, 2], dtype='int64')
    assert all(result == expected_result)

    a3 = tnp.array([1, 2, 3])
    b3 = tnp.array([4, 5, 6])
    result3 = b3 - a3
    expected_result3 = tnp.array([3, 3, 3], dtype='int64')   
    assert all(result3 == expected_result3)

def test__mul__():

    # test classic multiplication
    a = tnp.array([1, 2, 3])
    result = a * 1
    expected_result = tnp.array([1, 2, 3], dtype='int64')
    assert all(result == expected_result)

    b = tnp.array([5, 10, 15])
    result = b * 1.0
    expected_result = tnp.array([ 5.,  10.,  15.])
    assert all(result == expected_result)


    a2 = tnp.array([1, 2, 3])
    b2 = tnp.array([4, 5, 6])
    result2 = a2 * b2
    expected_result2 = tnp.array([4, 10, 18], dtype='int64')   
    assert all(result2 == expected_result2)

    # test __rmul__
    c = tnp.array([1, 2, 3])
    result = 1 * c
    expected_result = tnp.array([1, 2, 3], dtype='int64')
    assert all(result == expected_result)

    a3 = tnp.array([1, 2, 3])
    b3 = tnp.array([4, 5, 6])
    result3 = b3 * a3
    expected_result3 = tnp.array([4, 10, 18], dtype='int64')   
    assert all(result3 == expected_result3)


def test__truediv__():

    a = tnp.array([5, 10, 15])
    result = a / 2
    expected_result = tnp.array([ 2.5,  5.,  7.5], dtype='float64')
    assert all(result == expected_result)

    a2 = tnp.array([5, 10, 15])
    result = a / 5.0
    expected_result = tnp.array([ 1.,  2.,  3.], dtype='float64')
    assert all(result == expected_result)

    a3 = tnp.array([5, 10, 15])
    b3 = tnp.array([2, 3, 4])
    result = a3 / b3
    expected_result = tnp.array([ 2.5,  3.3333333333333335,  3.75])
    assert all(result == expected_result)

    with pytest.raises(ValueError):
        a4 = tnp.array([5, 10, 15, 3])
        b4 = tnp.array([2, 3, 4])
        result = a4 / b4

    with pytest.raises(ZeroDivisionError):
        a5 = tnp.array([5, 10, 15, 3])
        a5 / 0


def test__floordiv__():

    a = tnp.array([1, 2, 3])
    result = a // 1
    expected_result = tnp.array([1, 2, 3], dtype='int64')
    assert all(result == expected_result)

    b = tnp.array([5, 10, 15])
    result = a // 1.0
    expected_result = tnp.array([ 1.,  2.,  3.])
    assert all(result == expected_result)

    a2 = tnp.array([4, 6, 6])
    b2 = tnp.array([1, 2, 3])
    result2 = a2 // b2
    expected_result2 = tnp.array([4, 3, 2], dtype='int64')   
    assert all(result2 == expected_result2)

    with pytest.raises(ZeroDivisionError):
         a4 = tnp.array([1, 2, 3])
         result = a4 // 0

    with pytest.raises(ValueError):
        a3 = tnp.array([4, 6, 6, 3])
        b3 = tnp.array([1, 2, 3])
        result2 = a3 // b3

def test__mod__():

    a = tnp.array([1, 2, 3])
    result = a % 2
    expected_result = tnp.array([1, 0, 1])
    assert all(result == expected_result)

    b = tnp.array([5, 10, 15])
    result = b % 2.0
    expected_result = tnp.array([ 1.,  0.,  1.])
    assert all(result == expected_result)

    a2 = tnp.array([4, 6, 6])
    b2 = tnp.array([1, 2, 3])
    result2 = a2 % b2
    expected_result2 = tnp.array([0, 0, 0], dtype='int64')   
    assert all(result2 == expected_result2)

    with pytest.raises(ValueError):
        a3 = tnp.array([4, 6, 6, 3])
        b3 = tnp.array([1, 2, 3])
        result2 = a3 % b3

def test__pow__():

    a = tnp.array([1, 2, 3])
    result = a ** 2
    expected_result = tnp.array([1, 4, 9], dtype='int64')
    assert all(result == expected_result)
    
    b = tnp.array([5, 10, 15])
    result = b ** 2.0
    expected_result = tnp.array([ 25.,  100.,  225.])
    assert all(result == expected_result)

    a2 = tnp.array([4, 6, 6])
    b2 = tnp.array([1, 2, 3])
    result2 = a2 ** b2
    expected_result2 = tnp.array([4, 36, 216], dtype='int64')   
    assert all(result2 == expected_result2)

    with pytest.raises(ValueError):
        a3 = tnp.array([4, 6, 6, 3])
        b3 = tnp.array([1, 2, 3])
        result2 = a3 ** b3

def test__iadd__():

    a = tnp.array([1, 2, 3])
    a += 5
    expected_result = tnp.array([6, 7, 8])
    assert all(a == expected_result)

    a1 = tnp.array([3, 2, 1])
    b1 = tnp.array([8, 9, 2])
    a1 += b1
    expected_result = tnp.array([11, 11, 3], dtype='int64')
    assert all(a1 == expected_result)

    with pytest.raises(ValueError):
        a2 = tnp.array([3, 2, 1, 1])
        b2 = tnp.array([8, 9, 2])
        a2 += b2

def test__isub__():

    a = tnp.array([1, 2, 3])
    a -= 5
    expected_result = tnp.array([-4, -3, -2])
    assert all(a == expected_result)

    a1 = tnp.array([3, 2, 1])
    b1 = tnp.array([8, 9, 2])
    a1 -= b1
    expected_result = tnp.array([-5, -7, -1], dtype='int64')
    assert all(a1 == expected_result)

    with pytest.raises(ValueError):
        a2 = tnp.array([3, 2, 1, 1])
        b2 = tnp.array([8, 9, 2])
        a2 -= b2


def test__imul__():

    a = tnp.array([1, 2, 3])
    a *= 5
    expected_result = tnp.array([5, 10, 15])
    assert all(a == expected_result)

    a1 = tnp.array([3, 2, 1])
    b1 = tnp.array([8, 9, 2])
    a1 *= b1
    expected_result = tnp.array([24, 18, 2], dtype='int64')
    assert all(a1 == expected_result)

    with pytest.raises(ValueError):
        a2 = tnp.array([3, 2, 1, 1])
        b2 = tnp.array([8, 9, 2])
        a2 *= b2


def test__ifloordiv__():

    a = tnp.array([5, 10, 15])
    a //= 5
    expected_result = tnp.array([1, 2, 3])
    assert all(a == expected_result)

    a1 = tnp.array([9, 10, 2])
    b1 = tnp.array([3, 2, 1])
    a1 //= b1
    expected_result = tnp.array([3, 5, 2], dtype='int64')
    assert all(a1 == expected_result)

    with pytest.raises(ValueError):
        a2 = tnp.array([3, 2, 1, 1])
        b2 = tnp.array([8, 9, 2])
        a2 //= b2

def test__ipow__():

    a = tnp.array([5, 10, 15])
    a **= 5
    expected_result = tnp.array([3125, 100000, 759375], dtype='int64')
    assert all(a == expected_result)

    a1 = tnp.array([9, 10, 2])
    b1 = tnp.array([3, 2, 1])
    a1 **= b1
    expected_result = tnp.array([729, 100, 2], dtype='int64')
    assert all(a1 == expected_result)

    with pytest.raises(ValueError):
        a2 = tnp.array([3, 2, 1, 1])
        b2 = tnp.array([8, 9, 2])
        a2 **= b2

def test_dtype():
    
    for shape in [(9, ), (9, 4), (9, 4, 5)]:
        for dtype in ['bool', 'int8', 'uint8', 'int16', 'uint16',
                      'int32', 'uint32', 'float32', 'float64']:
            a = np.empty(shape, dtype=dtype)
            b = tnp.empty(shape, dtype=dtype)
            assert a.shape == b.shape
            assert a.dtype == b.dtype
            assert a.itemsize == b.itemsize
    
    raises(TypeError, tnp.zeros, (9, ), 'blaa')
    
    assert tnp.array([1.0, 2.0]).dtype == 'float64'
    assert tnp.array([1, 2]).dtype == 'int64'


def test_reshape():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='int32')
    b = tnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='int32')
    
    for shape in [(2, 4), (4, 2), (2, 2, 2), (8,)]:
        a.shape = shape
        b.shape = shape
        assert a.shape == b.shape
        assert a.strides == b.strides
    
    a.shape = 2, 4
    b.shape = 2, 4
    
    # Test transpose
    assert b.T.shape == (4, 2)
    assert (a.T == b.T).all()
    assert (b.T.T == b).all()
    
    # Make non-contiguous versions
    a2 = a[:, 2:]
    b2 = b[:, 2:]
    
    # Test contiguous flag
    assert a.flags['C_CONTIGUOUS']
    assert not a2.flags['C_CONTIGUOUS']
    
    # Test base
    assert a2.base is a
    assert b2.base is b
    assert a2[:].base is a
    assert b2[:].base is b

    # Test reshape
    reshaped_a = a.reshape((4, 2))
    reshaped_b = b.reshape((4, 2))
    assert(reshaped_a == reshaped_b).all()
    
    # Fail
    with raises(ValueError):  # Invalid shape
        a.shape = (3, 3)
    with raises(ValueError):
        b.shape = (3, 3)
    with raises(AttributeError):  # Cannot reshape non-contiguous arrays
        a2.shape = 4,
    with raises(AttributeError):
        b2.shape = 4,


def test_from_and_to_numpy():
    # This also tests __array_interface__
    
    for dtype in ['float32', 'float64', 'int32', 'uint32', 'uint8', 'int8']:
        for data in [[1, 2, 3, 4, 5, 6, 7, 8],
                    [[1, 2], [3, 4], [5, 6], [7, 8]],
                    [[[1, 2], [3, 4]],[[5, 6], [7, 8]]],
                    ]:
                        
            # Convert from numpy, from tinynumpy, to numpy
            a1 = np.array(data, dtype)
            b1 = tnp.array(a1)
            b2 = tnp.array(b1)
            a2 = np.array(b2)
            
            # Check if its the same
            for c in [b1, b2, a2]:
                assert a1.shape == c.shape
                assert a1.dtype == c.dtype
                assert a1.strides == c.strides
                assert (a1 == c).all()
    
    # Also test using a numpy array as a buffer
    a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], 'float32')
    b = tnp.ndarray(a.shape, a.dtype, strides=a.strides, buffer=a.ravel())
    assert (a==b).all()
    
    # Test that is indeed same data
    a[0, 0] = 99
    assert (a==b).all()


def test_from_ctypes():
    
    for type, dtype in [(ctypes.c_int16, 'int16'), 
                        (ctypes.c_uint8, 'uint8'), 
                        (ctypes.c_float, 'float32'), 
                        (ctypes.c_double, 'float64')]:
        # Create ctypes array, possibly something that we get from a c lib
        buffer = (type*100)()
        
        # Create array!
        b = tnp.ndarray((4, 25), dtype, buffer=buffer)
        
        # Check that we can turn it into a numpy array
        a = np.array(b, copy=False)
        assert (a == b).all()
        assert a.dtype == dtype
        
        # Verify that both point to the same data
        assert a.__array_interface__['data'][0] == ctypes.addressof(buffer)
        assert b.__array_interface__['data'][0] == ctypes.addressof(buffer)
        
        # also verify offset in __array_interface__ here
        for a0, b0 in zip([a[2:], a[:, 10::2], a[1::2, 10:20:2]],
                          [b[2:], b[:, 10::2], b[1::2, 10:20:2]]):
            pa = a0.__array_interface__['data'][0]
            pb = b0.__array_interface__['data'][0]
            assert pa > ctypes.addressof(buffer)
            assert pa == pb


def test_from_bytes():
    skip('Need ndarray.frombytes or something')
    # Create bytes
    buffer = b'x' * 100
    
    # Create array!
    b = tnp.ndarray((4, 25), 'uint8', buffer=buffer)
    ptr = ctypes.cast(buffer, ctypes.c_void_p).value
    
    # Check that we can turn it into a numpy array
    a = np.array(b, copy=False)
    assert (a == b).all()
    
    # Verify readonly
    with raises(Exception):
        a[0, 0] = 1  
    with raises(Exception):
        b[0, 0] = 1  
    
    # Verify that both point to the same data
    assert a.__array_interface__['data'][0] == ptr
    assert b.__array_interface__['data'][0] == ptr
    
    # also verify offset in __array_interface__ here
    for a0, b0 in zip([a[2:], a[:, 10::2], a[1::2, 10:20:2]],
                        [b[2:], b[:, 10::2], b[1::2, 10:20:2]]):
        pa = a0.__array_interface__['data'][0]
        pb = b0.__array_interface__['data'][0]
        assert pa > ptr
        assert pa == pb


def test_creating_functions():
    
    # Test array
    b1 = tnp.array([[1, 2, 3], [4, 5, 6]])
    assert b1.shape == (2, 3)
    

def test_getitem():
     
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    b = tnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])


def test_get_step():
    # Test for C-contiguous arrays
    c_array = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    tnp_c_array = tnp.array([[1, 2, 3], [4, 5, 6]], order='C')
    assert tnp._get_step(c_array, order='C') == 1
    assert tnp._get_step(tnp_c_array, order='C') == 1

    # Test for F-contiguous arrays
    f_array = np.array([[1, 2, 3], [4, 5, 6]], order='F')
    tnp_f_array = tnp.array([[1, 2, 3], [4, 5, 6]], order='F')
    assert tnp._get_step(f_array, order='F') == 0
    assert tnp._get_step(tnp_f_array, order='F') == 0

    # Test for non-contiguous arrays
    nc_array = c_array[:, ::2]
    tnp_nc_array = tnp_c_array[:, ::2]
    assert tnp._get_step(nc_array) == 0
    assert tnp._get_step(tnp_nc_array) == 0

    # Test for non-contiguous arrays with Fortran order
    f_nc_array = f_array[::2, :]
    tnp_f_nc_array = tnp_f_array[::2, :]
    assert tnp._get_step(f_nc_array, order='F') == 0
    assert tnp._get_step(tnp_f_nc_array, order='F') == 0


def test_setitem_writeable():

    a = tnp.array([1, 2, 3])
    a[0] = 4
    expected_result = tnp.array([4, 2, 3, 4, 5], dtype='int64')
    assert all(a == expected_result)

    with pytest.raises(RuntimeError):
        a = tnp.array([1, 2, 3])
        a.flags = {'WRITEABLE': False}
        a[0] = 4
    
    with pytest.raises(ValueError):
        a = tnp.array([1, 2, 3])
        a.flags = {'WRITEBACKIFCOPY': True}
        

def test_asfortranarray():
    """test the asfortranarray function for tinynumpy"""

    a = tnp.array([[1, 2, 3], [4, 5, 6]])
    if a.ndim >= 1:
        b = tnp.asfortranarray(a)
        result_F = b.flags['F_CONTIGUOUS']
        result_C = b.flags['C_CONTIGUOUS']
    assert result_F == True
    assert result_C == False

    assert b.flags['OWNDATA'] == False
    assert b.flags['WRITEABLE'] == True
    assert b.flags['ALIGNED'] == True
    assert b.flags['WRITEBACKIFCOPY'] == False

    expected_data = tnp.array([[1, 2, 3], [4, 5, 6]])
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            assert b[i, j] == expected_data[i][j]

    b = tnp.array([1, 2, 3])
    if b.ndim <= 1:
        c = tnp.asfortranarray(b)
        result_F = c.flags['F_CONTIGUOUS']
        result_C = b.flags['C_CONTIGUOUS']
    assert result_F == True
    assert result_C == True

    assert b.flags['OWNDATA'] == True
    assert b.flags['WRITEABLE'] == True
    assert b.flags['ALIGNED'] == True
    assert b.flags['WRITEBACKIFCOPY'] == False


def test_transpose():
    """test the transpose function for tinynumpy"""

    # Test 1D array
    a = tnp.array([1, 2, 3], dtype='int32')
    result = a.transpose()
    assert result.shape == a.shape
    assert (result == a).all()

    # Test 2D array
    b = tnp.array([[1, 2], [3, 4], [5, 6]], dtype='int32')
    result = b.transpose()
    result.shape == 2, 3
    expected_result = tnp.array([[1, 3, 5],[2, 4, 6]])
    assert result == expected_result

    # Test 3D array
    b = tnp.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype='int32')
    result = b.transpose()
    result.shape = 2, 2, 2
    expected_result = tnp.array([[[1, 3], [5, 7]], [[2, 4], [6, 8]]], dtype='int32')
    assert result == expected_result

    # Test when ndim > 3
    d = tnp.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype='int32')
    with pytest.raises(ValueError):
        result = d.transpose()

def test_squeeze_strides():
    """test the squeeze_strides function for tinynumpy"""
    a = tnp.array([[[0], [1], [2]]])
    result = tnp.squeeze_strides(a)
    expected_result = (tnp.array([[0],[1],[2]], dtype='int64'),)
    assert result == expected_result


def test__array_interface__():
    """test __array_interface__ for tinynumpy"""
    a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], 'float32')
    b = tnp.ndarray(a.shape, a.dtype, strides=a.strides, buffer=a.ravel())

    # __array_interface__ properties
    a_interface = a.__array_interface__
    b_interface = b.__array_interface__
    assert a_interface['typestr'] == b_interface['typestr']
    assert a_interface['shape'] == b_interface['shape']
    assert a_interface['data'][0] == b_interface['data'][0]
    assert a_interface['data'][1] == b_interface['data'][1]


# Start vector cross product tests
def test_cross():
    """test the cross product of two 3 dimensional vectors"""

    # Vector cross-product.
    x = [1, 2, 3]
    y = [4, 5, 6]
    z = tnp.cross(x, y)

    assert z == [-3, 6, -3]


def test_2dim_cross():
    """test the cross product of two 2 dimensional vectors"""

    # 2 dim cross-product.
    x = [1, 2]
    y = [4, 5]
    z = tnp.cross(x, y)

    assert z == [-3]


def test_4dim_cross():
    """test the cross product of two 2 dimensional vectors"""

    # 4 dim cross-product.
    x = [1, 2, 3, 4]
    y = [5, 6, 7, 8]

    with pytest.raises(IndexError) as execinfo:
        z = tnp.cross(x, y)

    assert 'Vector has invalid dimensions' in str(execinfo.value)


def test_mdim_cross():
    """test the cross product of two 4 dimensional vectors"""

    # Mixed dim cross-product.
    x = [1, 2, 3]
    y = [4, 5, 6, 7]

    with pytest.raises(IndexError) as execinfo:
        z = tnp.cross(x, y)

    assert 'Vector has invalid dimensions' in str(execinfo.value)


# Start vector dot product tests
def test_dot():
    """test the dot product of two mixed dimensional vectors"""

    # Vector dot-product.
    x = [1, 2, 3]
    y = [4, 5, 6]
    z = tnp.dot(x, y)

    assert z == 32


def test_2dim_dot():
    """test the dot product of two 2 dimensional vectors"""

    # 2 dim dot-product.
    x = [1, 2]
    y = [4, 5]
    z = tnp.dot(x, y)

    assert z == 14


def test_4dim_dot():
    """test the dot product of two 4 dimensional vectors"""

    # 4 dim dot-product.
    x = [1, 2, 3, 4]
    y = [5, 6, 7, 8]
    z = tnp.dot(x, y)

    assert z == 70


def test_mdim_dot():
    """test the dot product of two mixed dimensional vectors"""

    # Mixed dim dot-product.
    x = [1, 2, 3]
    y = [4, 5, 6, 7]

    with pytest.raises(IndexError) as execinfo:
        z = tnp.dot(x, y)

    assert 'Vector has invalid dimensions' in str(execinfo.value)


# Start vector determinant tests
def test_det():
    """test calculation of the determinant of a three dimensional"""

    # Three dim determinant
    x = [5, -2, 1]
    y = [0, 3, -1]
    z = [2, 0, 7]
    mat = [x, y, z]

    a = tnp.linalg.det(mat)

    assert a == 103

# Start simple math function tests
def test_add():
    """test the addition function for tinynumpy"""

    x = [5, -2, 1]
    y = [0, 3, -1]

    a = tnp.add(x,y)

    assert a == tnp.array([5, 1, 0], dtype='int64')


def test_subtract():
    """test the addition function for tinynumpy"""

    x = [5, -2, 1]
    y = [0, 3, -1]

    a = tnp.subtract(x,y)

    assert a == tnp.array([5, -5, -2], dtype='int64')


def test_divide():
    """test the addition function for tinynumpy"""

    x = [15, -12, 3]
    y = 3

    a = tnp.divide(x,y)

    assert a == tnp.array([5, -4, 1], dtype='int64')



def test_multiply():
    """test the addition function for tinynumpy"""

    x = [5, -2, 1]
    y = [0, 3, -1]

    a = tnp.multiply(x,y)

    assert a == tnp.array([0, -6, -1], dtype='int64')
    

def test_argwhere():
    """test the argwhere function for tinynumpy"""
    a = tnp.array([1,2,3,4,5])
    result = a.argwhere(3)

    expected_result = [[2]]
    assert result == expected_result


def test_tolist():
    """test the tolist function for tinynumpy"""

    a = tnp.array([1,2,3])
    result = a.tolist()

    expected_result = [1,2,3]
    assert result == expected_result
    

def test_repeat():
    """test the repeat function for tinynumpy"""
    a = tnp.array([1,2,3,4,5])
    result = a.repeat(3)

    expected_result = tnp.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype='int64')
    assert result == expected_result

    a = tnp.array([1,2,3,4,5])

    with pytest.raises(TypeError):
        result = a.repeat(1,1)

    with pytest.raises(ValueError):
        result = a.repeat(-1)


def test_zeros_like():
    """test the zeros_like function for tinynumpy"""
    a = tnp.array([[1,2,3],[4,5,6]])
    result = tnp.zeros_like(a)

    expected_result = tnp.array([[0, 0, 0],[0, 0, 0]], dtype='int64')
    assert result == expected_result


def test_ones_like():
    """test the ones_like function for tinynumpy"""
    a = tnp.array([[1,2,3],[4,5,6]])
    result = tnp.ones_like(a)

    expected_result = tnp.array([[1, 1, 1,],[1, 1, 1]], dtype='int64')
    assert result == expected_result


def test_empty_like():
    """test the empty_like function for tinynumpy"""
    a = tnp.array([[1,2,3],[4,5,6]])
    result = tnp.empty_like(a)

    expected_result = tnp.array([[0, 0, 0],[0, 0, 0]], dtype='int64')
    assert result == expected_result


def test_ones():
    """test the ones function for tinynumpy"""
    a = [1, 2, 3]
    result = tnp.ones(a)

    expected_result = tnp.array([[[ 1.,  1.,  1.],[ 1.,  1.,  1.]]])
    assert result == expected_result

    with pytest.raises(AssertionError):
        a = tnp.array([1, 2, 3])
        result = tnp.ones(a)

def test_arange():
    """test the arange function for tinynumpy"""
    a = tnp.array([1])
    result = tnp.arange(a)

    expected_result = tnp.array([0.])
    assert result == expected_result

def test_clip():
    """test the clip function for tinynumpy"""
    a = tnp.array([1, 2, 3])
    result = a.clip(1,2)

    expected_result = tnp.array([1, 2, 3], dtype='int64')
    assert result == expected_result

    # values outside range
    a = tnp.array([0, 5, 10])
    result = a.clip(1, 2)

    expected_result = tnp.array([1, 2, 2], dtype='int64')
    assert result == expected_result

    # conver to different data type
    a = tnp.array([1.5, 2.5, 3.5])
    result = a.clip(1, 2)

    expected_result = tnp.array([1.5, 2, 2], dtype='float64')
    assert result == expected_result

    # out parameter
    a = tnp.array([0, 5, 10])
    out = tnp.empty((3,), dtype='int64')
    result = a.clip(1, 2, out=out)

    expected_result = tnp.array([1, 2, 2], dtype='int64')
    assert result is out
    assert result == expected_result

    # negative values
    a = tnp.array([-1, 0, 1])
    result = a.clip(0, 2)

    expected_result = tnp.array([0, 0, 1], dtype='int64')
    assert result == expected_result

    # floating point values
    a = tnp.array([1.5, 2.5, 3.5])
    result = a.clip(1, 2)

    expected_result = tnp.array([1.5, 2, 2], dtype='float64')
    assert result == expected_result

def test_linspace():
    """test the linspace function for tinynumpy"""
    # default behavior
    result = tnp.linspace(0, 1)
    expected_result = tnp.array([ 0.,  0.02040816326530612,  0.04081632653061224,  0.061224489795918366,  0.08163265306122448,  0.1020408163265306,  0.12244897959183673,  0.14285714285714285,  0.16326530612244897,  0.18367346938775508,  0.2040816326530612,  0.22448979591836732,  0.24489795918367346,  0.26530612244897955,  0.2857142857142857,  0.3061224489795918,  0.32653061224489793,  0.3469387755102041,  0.36734693877551017,  0.3877551020408163,  0.4081632653061224,  0.42857142857142855,  0.44897959183673464,  0.4693877551020408,  0.4897959183673469,  0.5102040816326531,  0.5306122448979591,  0.5510204081632653,  0.5714285714285714,  0.5918367346938775,  0.6122448979591836,  0.6326530612244897,  0.6530612244897959,  0.673469387755102,  0.6938775510204082,  0.7142857142857142,  0.7346938775510203,  0.7551020408163265,  0.7755102040816326,  0.7959183673469387,  0.8163265306122448,  0.836734693877551,  0.8571428571428571,  0.8775510204081632,  0.8979591836734693,  0.9183673469387754,  0.9387755102040816,  0.9591836734693877,  0.9795918367346939,  0.9999999999999999])    
    assert all(result[i] == expected_result[i] for i in range(len(result)))

    # edge case
    result = tnp.linspace(0, 10, num=2)
    expected_result = tnp.array([0.0, 10.0])
    assert all(result[i] == expected_result[i] for i in range(len(result)))

    # return step
    result, step = tnp.linspace(0, 1, retstep=True)
    assert step == 0.02040816326530612
    
    # data types
    result = tnp.linspace(0, 1, dtype='float64')
    assert result.dtype == 'float64'


def test_logspace():
    """test the logspace function for tinynumpy"""
    # default
    result = tnp.logspace(0, 3, 10)
    expected_result = tnp.array([1.0, 2.154434690031884, 4.641588833612778, 10.0, 21.544346900318832, 46.4158883361278, 100.0, 215.44346900318845, 464.15888336127773, 1000.0])
    assert all(result[i] == expected_result[i] for i in range(len(result)))

    # num
    result = tnp.logspace(2.0, 3.0, num=4)
    expected_result = tnp.array([100.0, 215.44346900318845, 464.15888336127773, 1000.0])
    assert all(result[i] == expected_result[i] for i in range(len(result)))

    # endpoint
    result = tnp.logspace(2.0, 3.0, num=4, endpoint=False)
    expected_result = tnp.array([100.0, 177.82794100389228, 316.22776601683796, 562.341325190349, 1000.0])
    assert all(result[i] == expected_result[i] for i in range(len(result)))

    # base
    result = tnp.logspace(2.0, 3.0, num=4, base=2.0)
    expected_result = tnp.array([4.0, 5.039684199579493, 6.3496042078727974, 8.0])
    assert all(result[i] == expected_result[i] for i in range(len(result)))
    
def test_astype():
    """test the astype function for tinynumpy"""
    for dtype in ['bool', 'int8', 'uint8', 'int16', 'uint16',
                      'int32', 'uint32', 'int64', 'uint64', 'float32', 'float64']:

        a = tnp.array([1, 2, 3])
        result = a.astype(dtype)
        
        expected_result_bool = tnp.array([True, True, True], dtype='bool')
        assert result == expected_result_bool

        expected_result_int8 = tnp.array([1, 2, 3], dtype='int8')
        assert result == expected_result_int8

        expected_result_uint8 = tnp.array([1, 2, 3], dtype='uint8')
        assert result == expected_result_uint8

        expected_result_int16 = tnp.array([1, 2, 3], dtype='int16')
        assert result == expected_result_int16

        expected_result_uint16 = tnp.array([1, 2, 3], dtype='uint16')
        assert result == expected_result_uint16

        expected_result_int32 = tnp.array([1, 2, 3], dtype='int32')
        assert result == expected_result_int32

        expected_result_uint32 = tnp.array([1, 2, 3], dtype='uint32')
        assert result == expected_result_uint32

        expected_result_int64 = tnp.array([1, 2, 3], dtype='int64')
        assert result == expected_result_int64

        expected_result_uint64 = tnp.array([1, 2, 3], dtype='uint64')
        assert result == expected_result_uint64

        expected_result_float32 = tnp.array([ 1.,  2.,  3.], dtype='float32')
        assert result == expected_result_float32

        expected_result_float64 = tnp.array([ 1.,  2.,  3.], dtype='float64')
        assert result == expected_result_float64