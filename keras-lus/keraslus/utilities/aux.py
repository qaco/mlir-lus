from keraslus.utilities import lit
import math

def padding_conv2d(i_hw, k_hw, dilations, strides, padding):
    if padding == lit.VALID:
        oh = math.ceil(float(i_hw[0] - (k_hw[0] - 1) * dilations[1]) / float(strides[1]))
        ow = math.ceil(float(i_hw[1] - (k_hw[1] - 1) * dilations[2]) / float(strides[2]))
    elif padding == lit.SAME:
        oh = math.ceil(float(i_hw[0]) / float(strides[1]))
        ow = math.ceil(float(i_hw[1]) / float(strides[2]))
    else:
        assert(False)
    return (abs(oh), abs(ow))

def padding_maxpooling2d(i_hw, p_hw, strides, padding):
    if padding == lit.VALID:
        nh = math.floor(float(i_hw[0] - p_hw[0]) / float(strides[0])) + 1
        nw = math.floor(float(i_hw[1] - p_hw[1]) / float(strides[1])) + 1
    elif padding == lit.SAME:
        nh = math.floor((i_hw[0] - 1) / strides[0]) + 1
        nw = math.floor((i_hw[1] - 1) / strides[1]) + 1
    else:
        assert(False)
    return (nh, nw)

def to_int_pair(value):
    if type(value) == int:
        res = (value, value)
    elif ((type(value) == tuple or type(value) == list)
          and len(value)) == 2:
        res = value
    else:
        assert(False)
    return res

def to_int_square(value):
    if type(value) == int:
        res = [1, value, value, 1]
    elif type(value) == tuple and len(value) == 2:
        res = [1, value[0], value[1], 1]
    elif ((type(value) == tuple or type(value) == list)
          and len(value) == 4):
        res = [value[0], value[1], value[2], value[3]]
    else:
        assert(False)
    return res

from keraslus.simple_ops.reshape_ops import reshape

def to_ndims(x, ndims):
    assert(len(x.shape) <= ndims)
    while len(x.shape) != ndims:
        x = reshape(tshape = list((1,) + x.shape))(x)
    return x
        
