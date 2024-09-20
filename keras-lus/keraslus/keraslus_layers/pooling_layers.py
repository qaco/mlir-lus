from keraslus.utilities import lit
from keraslus.utilities.op_base import TfOp
from keraslus.simple_ops.constant import Constant
from keraslus.utilities import aux
from keraslus.simple_ops.reshape_ops import reshape

class Pooling2D(TfOp):
    def __init__(
            self,
            pool_size=(2, 2),
            strides=(1, 1),
            padding=lit.VALID,
            data_format=lit.NHWC, **kwargs
    ):
        self.pool_size = aux.to_int_pair(pool_size)
        self.strides = aux.to_int_pair(strides)
        self.padding = padding
        self.data_format = data_format
        super().__init__(init_shape=lit.POLY_SHAPE_4D)
        self.set_attr(lit.ATTR_KSIZE, "[1, " + str(self.pool_size[0]) + ", " + str(self.pool_size[1]) + ", 1]")
        self.set_attr(lit.ATTR_PAD, "\"" + self.padding + "\"")
        self.set_attr(lit.ATTR_STRIDES, "[1, " + str(self.strides[0]) + ", " + str(self.strides[1]) + ", 1]")
        self.set_attr(lit.ATTR_FORMAT, "\"" + data_format + "\"")

    def __call__(self, x):
        if (len(x.shape) == 3):
            x = reshape(tshape=[1, x.shape[0],
                                x.shape[1], x.shape[2]])(x)
        self.args = (x,)
        b, h, w, c = x.shape
        nh, nw = aux.padding_maxpooling2d((h,w), self.pool_size, self.strides,
                                          self.padding)
        self.res.shape = (b, nh, nw, c)
        return self.res

class MaxPooling2D(Pooling2D):
    def __init__(
            self,
            pool_size=(2, 2),
            strides=(1, 1),
            padding=lit.VALID,
            data_format=lit.NHWC, **kwargs
    ):
        self.mlir_name = "\"tf.MaxPool\""
        super().__init__(pool_size, strides, padding, data_format)
        self.set_attr(lit.ATTR_EXP_PAD, "[]")

class AveragePooling2D(Pooling2D):
    def __init__(self,
                 pool_size=(2,2),
                 strides=(2,2),
                 padding=lit.VALID,
                 data_format=lit.NHWC,
                 **kwargs):
        self.mlir_name = "\"tf.AvgPool\""
        super().__init__(pool_size, strides, padding, data_format)
        
class GlobalAveragePooling2D(TfOp):
    def __init__(self,
                 data_format=lit.NHWC,
                 keep_dims = "false", **kwargs):
        self.data_format = data_format
        self.mlir_name = "\"tf.Mean\""
        self.reduce_indices = Constant([1,2], "i32", (2,))
        super().__init__(init_shape=lit.POLY_SHAPE_2D)
        self.set_attr(lit.ATTR_FORMAT, "\"" + data_format + "\"")
        self.set_attr(lit.ATTR_KEEP_DIMS, keep_dims)

    def __call__(self, x):
        if (len(x.shape) == 3):
            x = reshape(tshape=[1, x.shape[0],
                                x.shape[1], x.shape[2]])(x)
        self.args = (x, self.reduce_indices.res)
        b, h, w, c = x.shape
        self.res.shape = (b,c)
        return self.res
