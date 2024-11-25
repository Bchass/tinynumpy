# -*- coding: utf-8 -*-
# Copyright (c) 2014, Almar Klein and Wade Brainerd
# tinynumpy is distributed under the terms of the MIT License.
#
# Original code by Wade Brainerd (https://github.com/wadetb/tinyndarray)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

""" 
A lightweight, pure Python, numpy compliant ndarray class.

The documenation in this module is rather compact. For details on each
function, see the corresponding documentation at:
http://docs.scipy.org/doc/numpy/reference/index.html Be aware that the
behavior of tinynumpy may deviate in some ways from numpy, or that
certain features may not be supported.
"""

from __future__ import division
from __future__ import absolute_import

import ctypes

from math import sqrt
from copy import copy, deepcopy
from collections.abc import Iterable
import operator

import tinynumpy.tinylinalg as linalg
from tinynumpy.tinylinalg import LinAlgError as LinAlgError

# Define version numer
__version__ = '0.0.1dev'

# Define dtypes: struct name, short name, numpy name, ctypes type
_dtypes = [('B', 'b1', 'bool', ctypes.c_bool),
           ('b', 'i1', 'int8', ctypes.c_int8),
           ('B', 'u1', 'uint8', ctypes.c_uint8),
           ('h', 'i2', 'int16', ctypes.c_int16),
           ('H', 'u2', 'uint16', ctypes.c_uint16),
           ('i', 'i4', 'int32', ctypes.c_int32),
           ('I', 'u4', 'uint32', ctypes.c_uint32),
           ('q', 'i8', 'int64', ctypes.c_int64),
           ('Q', 'u8', 'uint64', ctypes.c_uint64),
           ('f', 'f4', 'float32', ctypes.c_float),
           ('d', 'f8', 'float64', ctypes.c_double),
           ]

# Inject common dtype names
_known_dtypes = [d[2] for d in _dtypes]
for d in _known_dtypes:
    globals()[d] = d

newaxis = None

nan = float('nan')

def _convert_dtype(dtype, to='numpy'):
    """ Convert dtype, if could not find, pass as it was.
    """
    if dtype is None:
        return dtype
    dtype = str(dtype)
    index = {'array':0, 'short':1, 'numpy':2, 'ctypes':3}[to]
    for dd in _dtypes:
        if dtype in dd:
            return dd[index]
    return dtype  # Otherwise return original


def _ceildiv(a, b):
    return -(-a // b)


def _get_step(view, order='C'):
    """ Return step to walk over array. If 1, the array is fully
    C-contiguous. If 0, the striding is such that one cannot
    step through the array.
    """
    cont_strides = _strides_for_shape(view.shape, view.itemsize)
    
    if order == 'C':
        step = view.strides[-1] // cont_strides[-1]
    elif order == 'F':
        step = view.strides[0] // cont_strides[0]
    corrected_strides = tuple([i * step for i in cont_strides])
    
    almost_cont = view.strides == corrected_strides
    if almost_cont:
        return step
    else:
        return 0  # not contiguous

def _strides_for_shape(shape, itemsize, order='C'):
    strides = [0] * len(shape)
    if order == 'C':
        strides[-1] = itemsize
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
    elif order == 'F':
        strides[0] = itemsize
        for i in range(1, len(shape)):
            strides[i] = strides[i - 1] * shape[i - 1]
    return tuple(strides)



def _size_for_shape(shape):
    stride_product = 1
    for s in shape:
        stride_product *= s
    return stride_product


def squeeze_strides(s):
    """ Pop strides for singular dimensions. """
    return tuple([s[0]] + [s[i] for i in range(1, len(s)) if s[i] != s[i-1]])


def _shape_from_object(obj):
    def _shape_from_object_r(element, axis):
        if isinstance(element, list):
            for i, e in enumerate(element):
                _shape_from_object_r(e, axis + 1)
            while len(shape) <= axis:
                shape.append(0)
            shape[axis] = max(shape[axis], len(element))

    shape = []
    _shape_from_object_r(obj, 0)
    return tuple(shape)


def _assign_from_object(array, obj, order):
    def _assign_from_object_r(element, indices):
        if isinstance(element, list):
            for i, e in enumerate(element):
                new_indices = indices + [i]
                _assign_from_object_r(e, new_indices)
        else:
            if order == 'F':
                indices = indices[::1]
            array[tuple(indices)] = element

    _assign_from_object_r(obj, [])
    return array


def _increment_mutable_key(key, shape):
    for axis in reversed(range(len(shape))):
        key[axis] += 1
        if key[axis] < shape[axis]:
            return True
        if axis == 0:
            return False
        key[axis] = 0


def _key_for_index(index, shape):
    key = []
    cumshape = [1]
    for i in reversed(shape):
        cumshape.insert(0, cumshape[0] * i)
    for s in cumshape[1:-1]:
        n = index // s
        key.append(n)
        index -= n * s
    key.append(index)
    return tuple(key)


## Public functions


def array(obj, dtype=None, copy=True, order=None):
    """ array(obj, dtype=None, copy=True, order=None)
    
    Create a new array. If obj is an ndarray, and copy=False, a view
    of that array is returned. For details see:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html
    """
    dtype = _convert_dtype(dtype)
    
    if isinstance(obj, ndarray):
        # From existing array
        a = obj.view()
        if dtype is not None and dtype != a.dtype:
            a = a.astype(dtype)
        elif copy:
            a = a.copy()
        return a
    if hasattr(obj, '__array_interface__'):
        # From something that looks like an array, we can create
        # the ctypes array for this and use that as a buffer
        D = obj.__array_interface__
        # Get dtype
        dtype_orig = _convert_dtype(D['typestr'][1:])
        # Create array
        if D['strides']:
            itemsize = int(D['typestr'][-1])
            bufsize = D['strides'][0] * D['shape'][0] // itemsize
        else:
            bufsize = _size_for_shape(D['shape'])
        
        BufType = (_convert_dtype(dtype_orig, 'ctypes') * bufsize)
        buffer = BufType.from_address(D['data'][0])
        a = ndarray(D['shape'], dtype_orig,
                    buffer=buffer, strides=D['strides'], order=order)
        # Convert or copy?
        if dtype is not None and dtype != dtype_orig:
            a = a.astype(dtype)
        elif copy:
            a = a.copy()
        return a
    else:
        # From some kind of iterable
        shape = _shape_from_object(obj)
        # Try to derive dtype
        if dtype is None:
            el = obj
            while isinstance(el, (tuple, list)) and el:
                el = el[0]
            if isinstance(el, int):
                dtype = 'int64'
        if order is None:
            order = 'C'
        # Create array
        a = ndarray(shape, dtype, order=order)
        _assign_from_object(a, obj, order)
        return a


def zeros_like(a, dtype=None, order=None):
    """ Return an array of zeros with the same shape and type as a given array.
    """
    dtype = a.dtype if dtype is None else dtype
    order = 'C' if order is None else order
    return zeros(a.shape, dtype, order)


def ones_like(a, dtype=None, order=None):
    """ Return an array of ones with the same shape and type as a given array.
    """
    dtype = a.dtype if dtype is None else dtype
    order = 'C' if order is None else order
    return ones(a.shape, dtype, order)


def empty_like(a, dtype=None, order=None):
    """ Return a new array with the same shape and type as a given array.
    """
    dtype = a.dtype if dtype is None else dtype
    order = 'C' if order is None else order
    return empty(a.shape, dtype, order)


def zeros(shape, dtype=None, order=None):
    """Return a new array of given shape and type, filled with zeros
    """
    order = 'C' if order is None else order
    return empty(shape, dtype, order)


def ones(shape, dtype=None, order=None):
    """Return a new array of given shape and type, filled with ones
    """
    order = 'C' if order is None else order
    a = empty(shape, dtype, order)
    a.fill(1)
    return a


def eye(size):
    """Return a new 2d array with given dimensions, filled with ones on the
    diagonal and zeros elsewhere.
    """
    a = zeros((size,size))
    for i in range(size):
        a[i,i] = 1
    return a


def empty(shape, dtype=None, order=None):
    """Return a new array of given shape and type, without initializing entries
    """
    order = 'C' if order is None else order
    return ndarray(shape, dtype, order=order)


def arange(*args, **kwargs):
    """ arange([start,] stop[, step,], dtype=None)

    Return evenly spaced values within a given interval.
    
    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
    but returns an ndarray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use ``linspace`` for these cases.
    """
    # Get dtype
    dtype = kwargs.pop('dtype', None)
    if kwargs:
        x = list(kwargs.keys())[0]
        raise TypeError('arange() got an unexpected keyword argument %r' % x)
    # Parse start, stop, step
    if len(args) == 0:
        raise TypeError('Required argument "start" not found')
    elif len(args) == 1:
        start, stop, step = 0, int(args[0]), 1
    elif len(args) == 2:
        start, stop, step = int(args[0]), int(args[1]), 1
    elif len(args) == 3:
        start, stop, step = int(args[0]), int(args[1]), int(args[2])
    else:
        raise TypeError('Too many input arguments')
    # Init
    iter = range(start, stop, step)
    a = empty((len(iter),), dtype=dtype)
    a[:] = list(iter)
    return a


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
    """ linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    
    Return evenly spaced numbers over a specified interval. Returns num
    evenly spaced samples, calculated over the interval [start, stop].
    The endpoint of the interval can optionally be excluded.
    """
    # Prepare
    start, stop = float(start), float(stop)
    ra = stop - start
    if endpoint:
        step = ra / (num-1)
    else:
        step = ra / num
    # Create
    a = empty((num,), dtype)
    a[:] = [start + i * step for i in range(num)]
    # Return
    if retstep:
        return a, step
    else:
        return a

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=None):
    """ logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=None)

    Return numbers spaced evenly on a log scale.

    """

    start, stop = float(start), float(stop)
    ra = stop - start
    
    if endpoint:
        num -= 1

    a = empty((num + 1,), dtype)

    if isinstance(base, list):
        for i, b in enumerate(base):
            a = empty((num + 1,), dtype)
            a[:] = [b ** (start + i * ra / (num)) for i in range(num + 1)]
    else:
        a[:] = [base ** (start + i * ra / (num)) for i in range(num + 1)]

    return a


def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    """ meshgrid(*xi, copy=True, sparse=False, indexing='xy')

    Return a tuple of coordinate matrices from coordinate vectors.

    """

    ndim = len(xi)

    if indexing not in {'xy', 'ij'}:
        raise ValueError("Indexing must be 'xy' or 'ij'")
    
    # Adjust the order of inputs for 'xy' indexing
    if indexing == 'xy' and ndim > 1:
        xi = (xi[1], xi[0]) + xi[2:]

    # Get the lengths of each input array
    shapes = [len(x) for x in xi]

    if len(shapes) < 2:
        raise ValueError("At least two input arrays are required")

    # Create the output grids
    grids = []
    if not sparse:
        for i, x in enumerate(xi):
            if i == 0:
                # Repeat for columns (x-axis direction)
                grid = [list(x) for _ in range(shapes[1])]
                print(grid)
            else:
                # Repeat for rows (y-axis direction)
                grid = [[x_val] * shapes[0] for x_val in x]
            grids.append(array(grid))
    else:
        for i, x in enumerate(xi):
            shape = [1] * ndim
            shape[i] = len(x)
            grids.append(array(x))

    # Swap back grids if 'xy' indexing
    if indexing == 'xy' and ndim >= 2:
        grids[0], grids[1] = grids[1], grids[0]

    return tuple(grids)


def add(ndarray_vec1, ndarray_vec2):
    c = []
    for a, b in zip(ndarray_vec1, ndarray_vec2):
        c.append(a+b)
    cRay = array(c)
    return cRay

def subtract(ndarray_vec1, ndarray_vec2):
    c = []
    for a, b in zip(ndarray_vec1, ndarray_vec2):
        c.append(a-b)
    cRay = array(c)
    return cRay

def multiply(ndarray_vec1, ndarray_vec2):
    c = []
    for a, b in zip(ndarray_vec1, ndarray_vec2):
        c.append(a*b)
    cRay = array(c)
    return cRay

def divide(ndarray_vec1, integer):
    c = []
    for a in ndarray_vec1:
        c.append(a / integer)
    cRay = array(c)
    return cRay

def cross(u, v):
    """
    Return the cross product of two 2 or 3 dimensional vectors.
    """

    uDim = len(u)
    vDim = len(v)

    uxv = []

    # http://mathworld.wolfram.com/CrossProduct.html
    if uDim == vDim == 2:
        try:
            uxv = [u[0]*v[1]-u[1]*v[0]]            
        except LinAlgError as e:
            uxv = e        
    elif uDim == vDim == 3:
        try:
            for i in range(uDim):
                uxv = [u[1]*v[2]-u[2]*v[1], -(u[0]*v[2]-u[2]*v[0]),
                       u[0]*v[1]-u[1]*v[0]]
        except LinAlgError as e:
            uxv = e
    else:
        raise IndexError('Vector has invalid dimensions')
    return uxv

def dot(u, v):
    """
    Return the dot product of two equal-dimensional vectors.
    """

    uDim = len(u)
    vDim = len(v)

    # http://reference.wolfram.com/language/ref/Dot.html
    if uDim == vDim:
        try:
            u_dot_v = sum(map(operator.mul, u, v))
        except LinAlgError as e:
            u_dot_v = e
    else:
        raise IndexError('Vector has invalid dimensions')
    return u_dot_v

def reshape(X,shape):
    """
    Returns the reshaped image of an ndarray
    """
    assert isinstance(X, ndarray)
    assert isinstance(shape, tuple) or isinstance(shape, list)
    return X.reshape(shape)


def asfortranarray(self):
    """
    Convert the array to F-contiguous order.
    
    Returns:
        ndarray: A new array in F-contiguous order.

    """

    # calculate new strides
    strides = _strides_for_shape(self.shape, self._itemsize)

    # create new object with the same data from buffer
    out =  ndarray(self._shape, dtype=self._dtype, buffer=self._data,
                            offset=self._offset, strides=strides)
    out._asfortranarray = True

    return out


def sqrt(x):
    """
    Returns:
        ndarry: Array with dtype of float64

        list: List with sqrt applied

    """ 
    
    if isinstance(x, ndarray):
        # create arr of same shape and dtype
        out = empty(x.shape, dtype="float64")
        # apply sqrt
        out._data[:] = [value**0.5 if value >= 0 else nan for value in x._toflatlist()]
        return out
    elif isinstance(x, (int, float)):
        return x**0.5 if x >= 0 else float('nan')
    # list
    elif isinstance(x, list):
        return [sqrt(i) if isinstance(i, list) else str(i**0.5).rstrip('0') if i >= 0 else 'nan' for i in x]
    else:
        raise TypeError("Unsupported type for sqrt")
    


class ndarray(object):
    """ ndarray(shape, dtype='float64', buffer=None, offset=0,
                strides=None, order=None)
    
    Array class similar to numpy's ndarray, implemented in pure Python.
    This class can be distinguished from a real numpy array in that
    the repr always shows the dtype as a string, and for larger arrays
    (more than 100 elements) it shows a short one-line repr.
    
    An array object represents a multidimensional, homogeneous array
    of fixed-size items.  An associated data-type property describes the
    format of each element in the array.
    
    Arrays should be constructed using `array`, `zeros` or `empty` (refer
    to the See Also section below).  The parameters given here refer to
    a low-level method (`ndarray(...)`) for instantiating an array.
    
    Parameters
    ----------
    shape : tuple of ints
        Shape of created array.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.
    buffer : object contaning data, optional
        Used to fill the array with data. If another ndarray is given,
        the underlying data is used. Can also be a ctypes.Array or any
        object that exposes the buffer interface.
    offset : int, optional
        Offset of array data in buffer.
    strides : tuple of ints, optional
        Strides of data in memory.
    order : {'C', 'F'}, optional
        Row-major or column-major order.

    Attributes
    ----------
    T : ndarray
        Transpose of the array. In tinynumpy only supported for ndim <= 3.
    data : buffer
        The array's elements, in memory. In tinynumpy this is a ctypes array.
    dtype : str
        Describes the format of the elements in the array. In tinynumpy
        this is a string.
    flags : dict
        Dictionary containing information related to memory use, e.g.,
        'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.
    flat : iterator object
        Flattened version of the array as an iterator. In tinynumpy
        the iterator cannot be indexed.
    size : int
        Number of elements in the array.
    itemsize : int
        The memory use of each array element in bytes.
    nbytes : int
        The total number of bytes required to store the array data,
        i.e., ``itemsize * size``.
    ndim : int
        The array's number of dimensions.
    shape : tuple of ints
        Shape of the array.
    strides : tuple of ints
        The step-size required to move from one element to the next in
        memory. For example, a contiguous ``(3, 4)`` array of type
        ``int16`` in C-order has strides ``(8, 2)``.  This implies that
        to move from element to element in memory requires jumps of 2 bytes.
        To move from row-to-row, one needs to jump 8 bytes at a time
        (``2 * 4``).
    base : ndarray
        If the array is a view into another array, that array is its `base`
        (unless that array is also a view).  The `base` array is where the
        array data is actually stored.
    __array_interface__ : dict
        Dictionary with low level array information. Used by numpy to
        turn into a real numpy array. Can also be used to give C libraries
        access to the data via ctypes.
    
    See Also
    --------
    array : Construct an array.
    zeros : Create an array, each element of which is zero.
    empty : Create an array, but leave its allocated memory unchanged (i.e.,
            it contains "garbage").
    
    Notes
    -----
    There are two modes of creating an array:

    1. If `buffer` is None, then only `shape`, `dtype`, and `order`
       are used.
    2. If `buffer` is an object exposing the buffer interface, then
       all keywords are interpreted.
    
    """
    
    __slots__ = ['_dtype', '_shape', '_strides', '_itemsize', 
                 '_offset', '_base', '_data', '_flags_bool', '_asfortranarray']
    
    def __init__(self, shape, dtype='float64', buffer=None, offset=0,
                 strides=None, order='C'):

        # Check and set shape
        try : 
            assert isinstance(shape, Iterable)
            shape = tuple(shape)
        except Exception as e:
            raise AssertionError('The shape must be tuple or list')
        assert all([isinstance(x, int) for x in shape])
        self._shape = shape
        
        # Check and set dtype
        dtype = _convert_dtype(dtype) if (dtype is not None) else 'float64'
        if dtype not in _known_dtypes:
            raise TypeError('data type %r not understood' % dtype)
        self._dtype = dtype
        # Itemsize is directly derived from dtype
        self._itemsize = int(_convert_dtype(dtype, 'short')[-1])
        
        if buffer is None:
            # New array
            self._base = None
            # Check and set offset and strides
            assert offset == 0
            self._offset = 0
            # Set flag to true by default
            self._flags_bool = True
            # Check order
            if order == 'C':
                strides = _strides_for_shape(shape, self._itemsize, order='C')
            elif order == 'F':
                strides = _strides_for_shape(shape, self._itemsize, order='F')
            self._strides = strides
            self.flags = {
                'C_CONTIGUOUS': (order == 'C' or self.ndim <= 1),
                'F_CONTIGUOUS': (order == 'F' or self.ndim <= 1)
            }
        else:
            # Existing array
            if isinstance(buffer, ndarray) and buffer.base is not None:
                buffer = buffer.base
            # Keep a reference to e memory cleanup
            self._base = buffer
            # WRITEABLE should be True when creating a view
            self._flags_bool = True
            # Check to keep track of asfortranarray() and @property flag
            self._asfortranarray = False
            # for ndarray we use the data property
            if isinstance(buffer, ndarray):
                buffer = buffer.data
            # Check and set offset
            assert isinstance(offset, int) and offset >= 0
            self._offset = offset
            # Check and set strides
            if strides is None:
                strides = _strides_for_shape(shape, self.itemsize)
            assert isinstance(strides, tuple)
            assert all([isinstance(x, int) for x in strides])
            assert len(strides) == len(shape)
            self._strides = strides

        # If order is F we need to loop
        if order == 'F':
            total_elements = 1
            for dim in shape:
                total_elements += dim
            buffersize = total_elements
            BufferClass = _convert_dtype(dtype, 'ctypes') * buffersize
        else:
            buffersize = self._strides[0] * self._shape[0] // self._itemsize
            buffersize += self._offset
            BufferClass = _convert_dtype(dtype, 'ctypes') * buffersize
        # Create buffer
        if buffer is None:
            self._data = BufferClass()
        elif isinstance(buffer, ctypes.Array):
            self._data = BufferClass.from_address(ctypes.addressof(buffer))
        else:
            self._data = BufferClass.from_buffer(buffer)
    
    @property
    def __array_interface__(self):
        """ Allow converting to real numpy array, or pass pointer to C library
        http://docs.scipy.org/doc/numpy/reference/arrays.interface.html
        """
        readonly = False
        # typestr
        typestr = '<' + _convert_dtype(self.dtype, 'short')
        # Pointer
        if isinstance(self._data, ctypes.Array):
            ptr = ctypes.addressof(self._data)
        elif hasattr(self._data, '__array_interface__'):
            ptr, readonly = self._data.__array_interface__['data']
        elif hasattr(self._data, 'buffer_info'):  # Python's array.array
            ptr = self._data.buffer_info()[0]
        elif isinstance(self._data, bytes):
            ptr = ctypes.cast(self._data, ctypes.c_void_p).value
            readonly = True
        else:
            raise TypeError('Cannot get address to underlying array data')
        ptr += self._offset * self.itemsize
        #
        return dict(version=3,
                    shape=self.shape,
                    typestr=typestr,
                    descr=[('', typestr)],
                    data=(ptr, readonly),
                    strides=self.strides,
                    #offset=self._offset,
                    #mask=None,
                    )
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, key):
        offset, shape, strides = self._index_helper(key)
        if not shape:
            # Return scalar
            return self._data[offset]
        else:
            # Return view
            return ndarray(shape, self.dtype,
                           offset=offset, strides=strides, buffer=self)
    
    def __setitem__(self, key, value):
        
        # Get info for view
        offset, shape, strides = self._index_helper(key)

        # Check if flag is True or False
        if not self._flags_bool:
            raise RuntimeError ("Array is not writeable")
        else:
            # Is this easy?
            if not shape:
                self._data[offset] = value
                return

        # Create view to set data to
        view = ndarray(shape, self.dtype,
                        offset=offset, strides=strides, buffer=self)
        
        # Get data to set as a list (because getting slices from ctype
        # arrays yield lists anyway). The list is our "contiguous array" 
        if isinstance(value, (float, int)):
            value_list = [value] * view.size
        elif isinstance(value, (tuple, list)):
            value_list = value
        else:
            if not isinstance(value, ndarray):
                value = array(value, copy=False)
            value_list = value._toflatlist()
        
        # Check if size match
        if view.size != len(value_list):
            raise ValueError('Number of elements in source does not match '
                                'number of elements in target.')
        
        # Assign data in most efficient way that we can. This code
        # looks for the largest semi-contiguous block: the block that
        # we can access as a 1D array with a stepsize.
        subviews = [view]
        value_index = 0
        count = 0
        while subviews:
            subview = subviews.pop(0)
            step = _get_step(subview)
            if step:
                block = value_list[value_index:value_index+subview.size]
                s = slice(subview._offset, 
                            subview._offset + subview.size * step, 
                            step)
                view._data[s] = block
                value_index += subview.size
                count += 1
            else:
                for i in range(subview.shape[0]):
                    subviews.append(subview[i])
        assert value_index == len(value_list)
    
    def __float__(self):
        if self.size == 1:
            return float(self.data[self._offset])
        else:
            raise TypeError('Only length-1 arrays can be converted to scalar')
    
    def __int__(self):
        if self.size == 1:
            return int(self.data[self._offset])
        else:
            raise TypeError('Only length-1 arrays can be converted to scalar')
    
    def _repr_r(self, s, axis, offset):
        axisindent = min(2, max(0, (self.ndim - axis - 1)))
        if axis < len(self._shape):
            s += '['
            for k_index in range(self._shape[axis]):
                if k_index > 0:
                    s += ('\n       ' + ' ' * axis) * axisindent
                if axis == self.ndim - 1:  # Last axis
                    offset_ = offset + k_index * self._strides[axis] // self._itemsize
                    elem_repr = repr(self._data[offset_])
                    if self._dtype.startswith('float'):
                        if elem_repr.endswith('.0'):
                            elem_repr = elem_repr[:-2]  # Remove trailing '.0'
                    s += elem_repr
                else:
                    offset_ = offset + k_index * self._strides[axis] // self._itemsize
                    s = self._repr_r(s, axis + 1, offset_)
                if k_index < self._shape[axis] - 1:
                    s += ', '
            s += ']'
        return s

    def __repr__(self):
        # If more than 100 elements, show short repr
        if self.size > 100:
            shapestr = 'x'.join(str(i) for i in self._shape)
            return f'<ndarray {shapestr} {self._dtype} at 0x{id(self):x}>'
        
        # Otherwise, try to show in nice way
        s = self._repr_r('', 0, self._offset)
        if self._dtype not in {'float64', 'int32'}:
            return f"array({s}, dtype='{self._dtype}')"
        else:
            return f"array({s})"
    
    def __eq__(self, other):
        if other.__module__.split('.')[0] == 'numpy':
            return other == self
        else:
            out = empty(self.shape, 'bool')
            out[:] = [i1==i2 for (i1, i2) in zip(self.flat, other.flat)]
            return out
    
    def __add__(self, other):
        '''classic addition
        '''
        if isinstance(other, (int, float)):
            out_dtype = 'float64' if isinstance(other, float) else self.dtype
            out = empty(self.shape, out_dtype)
            out[:] = [dat+other for dat in self._data]
            return out
        if isinstance(other, ndarray):
            if self.shape == other.shape :
                out = empty(self.shape, self.dtype)
                out[:] = [i+j for (i,j) in zip(self.flat, other.flat)]
                return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            out_dtype = 'float64' if isinstance(other, float) else self.dtype
            out = empty(self.shape, out_dtype)
            out[:] = [dat-other for dat in self._data] 
            return out
        if isinstance(other, ndarray):
            if self.shape == other.shape :
                out = empty(self.shape, self.dtype)
                out[:] = [i-j for (i,j) in zip(self.flat, other.flat)]
                return out

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        '''multiply element-wise with array or float/scalar'''
        if isinstance(other, (int, float)):
            out_dtype = 'float64' if isinstance(other, float) else self.dtype
            out = empty(self.shape, out_dtype)
            out[:] = [dat*other for dat in self._data] 
            return out
        if isinstance(other, ndarray):
            if self.shape == other.shape :
                out = empty(self.shape, self.dtype)
                out[:] = [i*j for (i,j) in zip(self.flat, other.flat)]
                return out       

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        '''divide element-wise with array or float/scalar'''
        if isinstance(other, (int, float)):
            if other == 0 : raise ZeroDivisionError
            out = empty(self.shape, 'float64')
            for i, dat in enumerate(self._data):
                out[i] = dat / other
            return out
        if isinstance(other, ndarray):
            if self.shape == other.shape :
                out = empty(self.shape, 'float64')
                out[:] = [i/j for (i,j) in zip(self.flat, other.flat)]
                return out
            else :
                raise ValueError('Array sizes do not match. '+str(self.shape)\
                                                  +' versus '+str(other.shape))

    def __floordiv__(self, other):
        '''divide element-wise with array or float/scalar'''
        if isinstance(other, (int, float)):
            out_dtype = 'float64' if isinstance(other, float) else self.dtype
            if other == 0 : raise ZeroDivisionError
            out = empty(self.shape, 'float64')
            out[:] = [dat//other for dat in self._data] 
            return out
        if (isinstance(other, ndarray)):
            if self.shape == other.shape :
                out = empty(self.shape, 'float64')
                out[:] = [i//j for (i,j) in zip(self.flat, other.flat)]
                return out
            else :
                raise ValueError('Array sizes do not match. '+str(self.shape)\
                                                  +' versus '+str(other.shape))

    def __mod__(self, other):
        '''divide element-wise with array or float/scalar'''
        if isinstance(other, (int, float)):
            out_dtype = 'float64' if isinstance(other, float) else self.dtype
            out = empty(self.shape, out_dtype)
            out[:] = [dat%other for dat in self._data] 
            return out
        if isinstance(other, ndarray):
            if self.shape == other.shape :
                out = empty(self.shape, self.dtype)
                out[:] = [i%j for (i,j) in zip(self.flat, other.flat)]
                return out
            else :
                raise ValueError('Array sizes do not match. '+str(self.shape)\
                                                  +' versus '+str(other.shape))

    def __pow__(self, other):
        '''power of two arrays element-wise (of just float power)'''
        if isinstance(other, (int, float)):
            out_dtype = 'float64' if isinstance(other, float) else self.dtype
            out = empty(self.shape, out_dtype)
            out[:] = [dat**other for dat in self._data] 
            return out
        if isinstance(other, ndarray):
            if self.shape == other.shape :
                out = empty(self.shape, self.dtype)
                out[:] = [i**j for (i,j) in zip(self.flat, other.flat)]
                return out
            else :
                raise ValueError('Array sizes do not match. '+str(self.shape)\
                                                  +' versus '+str(other.shape))

    def __iadd__(self, other):
        '''Addition of other array or float in place with += operator
        '''
        if isinstance(other, (int, float)):
            for i in range(len(self._data)):
                self._data[i]+=other
            return self
        if (isinstance(other, ndarray)):
            if self.shape == other.shape :
                for i in range(len(self._data)):
                    self._data[i]+=other._data[i]
                return self            
            else :
                raise ValueError('Array sizes do not match. '+str(self.shape)\
                                                  +' versus '+str(other.shape))

    def __isub__(self, other):
        '''Addition of other array or float in place with += operator
        '''
        if (isinstance(other, int) or isinstance(other, float)) :
            for i in range(len(self._data)):
                self._data[i]-=other
            return self
        if (isinstance(other, ndarray)):
            if self.shape == other.shape :
                for i in range(len(self._data)):
                    self._data[i]-=other._data[i]
                return self
            else :
                raise ValueError('Array sizes do not match. '+str(self.shape)\
                                                  +' versus '+str(other.shape))

    def __imul__(self, other):
        '''multiplication woth other array or float in place with *= operator
        '''
        if (isinstance(other, int) or isinstance(other, float)) :
            for i in range(len(self._data)):
                self._data[i]*=other
            return self
        if (isinstance(other, ndarray)):
            if self.shape == other.shape :
                for i in range(len(self._data)):
                    self._data[i]*=other._data[i]
                return self
            else :
                raise ValueError('Array sizes do not match. '+str(self.shape)\
                                                  +' versus '+str(other.shape))


    def __itruediv__(self, other):
        '''Division of other array or float in place with /= operator
        '''
        if (isinstance(other, int) or isinstance(other, float)) :
            if other == 0 : raise ZeroDivisionError
            for i in range(len(self._data)):
                self._data[i]/=other
            return self
        if (isinstance(other, ndarray)):
            if self.shape == other.shape :
                for i in range(len(self._data)):
                    self._data[i]/=other._data[i]
                return self
            else :
                raise ValueError('Array sizes do not match. '+str(self.shape)\
                                                  +' versus '+str(other.shape))

    def __ifloordiv__(self, other):
        '''Division of other array or float in place with /= operator
        '''
        if (isinstance(other, int) or isinstance(other, float)) :
            if other == 0 : raise ZeroDivisionError
            for i in range(len(self._data)):
                self._data[i]//=other
            return self
        if (isinstance(other, ndarray)):
            if self.shape == other.shape :
                for i in range(len(self._data)):
                    self._data[i]//=other._data[i]
                return self
            else :
                raise ValueError('Array sizes do not match. '+str(self.shape)\
                                                  +' versus '+str(other.shape))

    def __imod__(self, other):
        '''mod of other array or float in place with /= operator
        '''
        if (isinstance(other, int) or isinstance(other, float)) :
            if other == 0 : raise ZeroDivisionError
            for i in range(len(self._data)):
                self._data[i]%=other
            return self
        if (isinstance(other, ndarray)):
            if self.shape == other.shape :
                for i in range(len(self._data)):
                    self._data[i]%=other._data[i]
                return self
            else :
                raise ValueError('Array sizes do not match. '+str(self.shape)\
                                                  +' versus '+str(other.shape))

    def __ipow__(self, other):
        '''mod of other array or float in place with /= operator
        '''
        if (isinstance(other, int) or isinstance(other, float)) :
            for i in range(len(self._data)):
                self._data[i]**=other
            return self
        if (isinstance(other, ndarray)):
            if self.shape == other.shape :
                for i in range(len(self._data)):
                    self._data[i]**=other._data[i]
                return self
            else :
                raise ValueError('Array sizes do not match. '+str(self.shape)\
                                                  +' versus '+str(other.shape))


    ## Private helper functions
    
    def _index_helper(self, key):
        
        # Indexing spec is located at:
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

        # Promote to tuple.
        if not isinstance(key, tuple):
            key = (key,)

        axis = 0
        shape = []
        strides = []
        offset = self._offset

        for k in key:
            axissize = self._shape[axis]
            if isinstance(k, int):
                if k >= axissize:
                    raise IndexError('index %i is out of bounds for axis %i '
                                     'with size %s' % (k, axis, axissize))
                offset += k * self._strides[axis] // self.itemsize
                axis += 1
            elif isinstance(k, slice):
                start, stop, step = k.indices(self.shape[axis])
                shape.append(_ceildiv(stop - start, step))
                strides.append(step * self._strides[axis])
                offset += start * self._strides[axis] // self.itemsize
                axis += 1
            elif k is Ellipsis:
                raise TypeError("ellipsis are not supported.")
            elif k is None:
                shape.append(1)
                stride = 1
                for s in self._strides[axis:]:
                    stride *= s
                strides.append(stride)
            else:
                raise TypeError("key elements must be instaces of int or slice.")

        shape.extend(self.shape[axis:])
        strides.extend(self._strides[axis:])
        
        return offset, tuple(shape), tuple(strides)
    
    def _toflatlist(self):
        value_list = []
        subviews = [self]
        count = 0
        while subviews:
            subview = subviews.pop(0)
            step = _get_step(subview)
            if step:
                s = slice(subview._offset, 
                          subview._offset + subview.size * step, 
                          step)
                value_list += self._data[s]
                count += 1
            else:
                for i in range(subview.shape[0]):
                    subviews.append(subview[i])
        return value_list
    
    ## Properties
    
    @property
    def ndim(self):
        return len(self._shape)
    
    @property
    def size(self):
        return _size_for_shape(self._shape)
    
    @property
    def nbytes(self):
        return _size_for_shape(self._shape) * self.itemsize
    
    def _get_shape(self):
        return self._shape
    
    def _set_shape(self, newshape):
        if newshape == self.shape:
            return
        if self.size != _size_for_shape(newshape):
            raise ValueError('Total size of new array must be unchanged')
        if _get_step(self) == 1:
            # Contiguous, hooray!
            self._shape = tuple(newshape)
            self._strides = _strides_for_shape(self._shape, self.itemsize)
            return
        
        # Else, try harder ... This code supports adding /removing
        # singleton dimensions. Although it may sometimes be possible
        # to split a dimension in two if the contiguous blocks allow
        # this, we don't bother with such complex cases for now.
        # Squeeze shape / strides 
        N = self.ndim
        shape = [self.shape[i] for i in range(N) if self.shape[i] > 1]
        strides = [self.strides[i] for i in range(N) if self.shape[i] > 1]
        # Check if squeezed shapes match
        newshape_ = [newshape[i] for i in range(len(newshape)) 
                     if newshape[i] > 1]
        if newshape_ != shape:
            raise AttributeError('incompatible shape for non-contiguous array')
        # Modify to make this data work in loop
        strides.append(strides[-1])
        shape.append(1)
        # Form new strides
        i = -1
        newstrides = []
        try:
            for s in reversed(newshape):
                if s == 1:
                    newstrides.append(strides[i] * shape[i])
                else:
                    i -= 1
                    newstrides.append(strides[i])
        except IndexError:
            # Fail
            raise AttributeError('incompatible shape for non-contiguous array')
        else:
            # Success
            newstrides.reverse()
            self._shape = tuple(newshape)
            self._strides = tuple(newstrides)
    
    shape = property(_get_shape, _set_shape)  # Python 2.5 compat (e.g. Jython)
    
    @property
    def strides(self):
        return self._strides
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def itemsize(self):
        return self._itemsize
    
    @property
    def base(self):
        return self._base
    
    @property
    def data(self):
        return self._data
    
    @property
    def flat(self):
        subviews = [self]
        count = 0
        while subviews:
            subview = subviews.pop(0)
            step = _get_step(subview)
            if step:
                s = slice(subview._offset, 
                          subview._offset + subview.size * step, 
                          step)
                for i in self._data[s]:
                    yield i
            else:
                for i in range(subview.shape[0]):
                    subviews.append(subview[i])
    
    @property
    def T(self):
        if self.ndim < 2:
            return self
        else:
            return self.transpose()
    
    @property
    def flags(self):
        c_cont = _get_step(self) == 1
        f_cont = _get_step(self) == 1
        return {'C_CONTIGUOUS': (c_cont and not self._asfortranarray),
                'F_CONTIGUOUS': (f_cont and self.ndim <=1 or self._asfortranarray),
                'OWNDATA': self._base is None,
                'WRITEABLE': self._flags_bool,
                'ALIGNED': True,
                'WRITEBACKIFCOPY': False}

    @flags.setter
    def flags(self, value):
        if isinstance(value, dict):
            if 'WRITEABLE' in value:
                self._flags_bool = value['WRITEABLE']
            if 'F_CONTIGUOUS' in value:
                self._asfortranarray = value['F_CONTIGUOUS']
            if 'C_CONTIGUOUS' in value:
                self._asfortranarray = not value['C_CONTIGUOUS']
            if 'WRITEBACKIFCOPY' in value and value['WRITEBACKIFCOPY'] == True:
                raise ValueError("can't set WRITEBACKIFCOPY to True")
    
    ## Methods - managemenet
    
    def fill(self, value):
        assert isinstance(value, (int, float))
        self[:] = value
    
    def clip(self, a_min, a_max, out=None):
        if out is None:
            out = empty(self.shape, self.dtype)
        L = self._toflatlist()
        L = [min(a_max, max(a_min, x)) for x in L]
        out[:] = L
        return out
    
    def copy(self):
        out = empty(self.shape, self.dtype)
        out[:] = self
        return out
    
    def flatten(self):
        out = empty((self.size,), self.dtype)
        out[:] = self
        return out
    
    def ravel(self):
        return self.reshape((self.size, ))
    
    def repeat(self, repeats, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        out = empty((self.size * repeats,), self.dtype)
        for i in range(repeats):
            out[i*self.size:(i+1)*self.size] = self
        return out
    
    def reshape(self, newshape):
        out = self.view()
        try:
            out.shape = newshape
        except AttributeError:
            out = self.copy()
            out.shape = newshape
        return out
    
    def transpose(self):

        ndim = self.ndim
        if ndim < 2:
            return self.view()
        shape = self.shape[::-1]
        out = empty(shape, self.dtype)

        if ndim == 2:
            for i in range(self.shape[0]):
                out[:, i] = self[i, :]
        elif ndim == 3:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    out[:, j, i] = self[i, j, :]
        else:
            raise ValueError('Tinynumpy supports transpose up to ndim=3')
        return out
    
    def astype(self, dtype):
        out = empty(self.shape, dtype)
        out[:] = self
        return out
    
    def view(self, dtype=None, type=None):
        if dtype is None:
            dtype = self.dtype
        if dtype == self.dtype:
            return ndarray(self.shape, dtype, buffer=self, 
                           offset=self._offset, strides=self.strides)
        elif self.ndim == 1:
            itemsize = int(_convert_dtype(dtype, 'short')[-1])
            size = self.nbytes // itemsize
            offsetinbytes = self._offset * self.itemsize
            offset = offsetinbytes // itemsize
            return ndarray((size, ), dtype, buffer=self, offset=offset)
        else:
            raise ValueError('new type not compatible with array.')
    
    ## Methods - statistics
    
    # We use the self.flat generator here. self._toflatlist() would be
    # faster, but it might take up significantly more memory.
    
    def all(self, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        return all(self.flat)
    
    def any(self, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        return any(self.flat)
    
    def min(self, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        return min(self.flat)
    
    def max(self, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        return max(self.flat)
        #return max(self._toflatlist())  # almost twice as fast
    
    def sum(self, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        return sum(self.flat)
    
    def prod(self, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        p = 1.0
        for i in self.flat:
            p *= float(i)
        return p
        
    def ptp(self, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        mn = self.data[self._offset]
        mx = mn
        for i in self.flat:
            if i > mx:
                mx = i
            if i < mn:
                mn = i
        return mx - mn

    def mean(self, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        return self.sum() / self.size
    
    def argmax(self, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        r = self.data[self._offset]
        r_index = 0
        for i_index, i in enumerate(self.flat):
            if i > r:
                r = i
                r_index = i_index
        return r_index

    def argmin(self, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        r = self.data[self._offset]
        r_index = 0
        for i_index, i in enumerate(self.flat):
            if i < r:
                r = i
                r_index = i_index
        return r_index
    
    def cumprod(self, axis=None, out=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        if out is None:
            out = empty((self.size,), self.dtype)
        p = 1
        L = []
        for x in self.flat:
            p *= x
            L.append(p)
        out[:] = L
        return out

    def cumsum(self, axis=None, out=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        if out is None:
            out = empty((self.size,), self.dtype)
        p = 0
        L = []
        for x in self.flat:
            p += x
            L.append(p)
        out[:] = L
        return out

    def var(self, axis=None):
        if axis:
            raise (TypeError, "axis argument is not supported")
        m = self.mean()
        acc = 0
        for x in self.flat:
            acc += abs(x - m) ** 2
        return acc / self.size

    def std(self, axis=None):
        return sqrt(self.var(axis))

    def argwhere(self, val):
        #assumes that list has only values of same dtype

        idx  = [i for i, e in enumerate(self.flat) if e == val]
        keys = [list(_key_for_index(i, self.shape)) for i in idx]
        return keys

    def tolist(self):
        '''
        Returns the ndarray as a comprehensive list 
        '''
        shp    = list(self.shape).copy()
        jump   = self.size//shp[-1]
        n_comp = 0 #comprehension depth
        comp   = list(self._data).copy()
        while n_comp < len(self.shape)-1 :
            comp = [comp[i*shp[-1]:i*shp[-1]+shp[-1]] for i in range(jump)]
            shp.pop()
            jump = len(comp)//shp[-1]
            n_comp +=1
        return comp


class nditer:
    def __init__(self, array):
        self.array = array
        self.key = [0] * len(self.array.shape)

    def __iter__(self):
        return self

    def __len__(self):
        return _size_for_shape(self.array.shape)

    def __getitem__(self, index):
        key = _key_for_index(index, self.array.shape)
        return self.array[key]

    def __next__(self):
        if self.key is None:
            raise StopIteration
        value = self.array[tuple(self.key)]
        if not _increment_mutable_key(self.key, self.array.shape):
            self.key = None
        return value

    def next(self):
        return self.__next__()