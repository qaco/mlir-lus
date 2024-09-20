from keraslus.utilities.op_base import TfOp
from keraslus.utilities import lit
from keraslus.utilities.weight import Weight

class Add(TfOp):
    def __init__(self, dtype=None, init_shape=None, **kwargs):
        self.mlir_name = "\"tf.AddV2\""
        super().__init__(dtype, init_shape)

    def __call__(self, xy):
        x = xy[0]
        y = xy[1]
        self.args = (x,y)
        self.res.shape = x.shape
        self.res.elt_type = x.elt_type
        return self.res

class Sub(TfOp):
    def __init__(self, dtype=None, init_shape=None, **kwargs):
        self.mlir_name = "\"tf.SubOp\""
        super().__init__(dtype, init_shape)

    def __call__(self, xy):
        x = xy[0]
        y = xy[1]
        self.args = (x,y)
        self.res.shape = x.shape
        self.res.elt_type = x.elt_type
        return self.res

class Mul(TfOp):
    def __init__(self, dtype=None, init_shape=None, **kwargs):
        self.mlir_name = "\"tf.Mul\""
        super().__init__(dtype, init_shape)

    def __call__(self, xy):
        x = xy[0]
        y = xy[1]
        self.args = (x,y)
        self.res.shape = x.shape
        self.res.elt_type = x.elt_type
        return self.res

class biasadd(TfOp):
    def __init__(self, dtype=None, init_shape=None, data_format=lit.NHWC,
                 **kwargs):
        self.mlir_name = "\"tf.BiasAdd\""
        super().__init__(dtype, init_shape)
        self.set_attr(lit.ATTR_FORMAT, "\"" + data_format + "\"")

    def __call__(self, x, bias):
        self.args = (x, bias)
        self.res.shape = x.shape
        self.res.elt_type = x.elt_type
        return self.res

class matmul(TfOp):

    def __init__(self, dtype=None, init_shape=None, **kwargs):
        self.mlir_name = "\"tf.MatMul\""
        super().__init__(dtype, init_shape)
        self.set_attr(lit.ATTR_TRANS_A, "false")
        self.set_attr(lit.ATTR_TRANS_B, "false")
    
    def __call__(self, x, y):
        self.args = (x,y)
        self.res.shape = (x.shape[0], y.shape[1])
        self.res.elt_type = x.elt_type
        return self.res

class LessEqual(TfOp):
    def __init__(self, **kwargs):
        self.mlir_name = "\"tf.LessEqual\""
        super().__init__("i1", ())

    def __call__(self, x, y):
        self.args = (x,y)
        return self.res

class Equal(TfOp):
    def __init__(self, **kwargs):
        self.mlir_name = "\"tf.Equal\""
        super().__init__("i1", ())

    def __call__(self, x, y):
        self.args = (x,y)
        return self.res

class FloorMod(TfOp):
    def __init__(self, dtype=None, init_shape=None, **kwargs):
        self.mlir_name = "\"tf.FloorMod\""
        super().__init__(dtype, init_shape)
    
    def __call__(self, x, y):
        self.args = (x,y)
        self.res.shape = x.shape
        self.res.elt_type = x.elt_type
        return self.res

class WithBias:
    def __init__(self, use_bias, bshape,
                 bias_initializer,
                 bias_regularizer, bias_constraint,
                 p_name,
                 k_name='bias:0'):
        self.use_bias = use_bias
        self.bias_shape = bshape
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
        if use_bias:
            self.bias_weight = Weight(value=None,
                                      p_name=p_name,
                                      k_name=k_name,
                                      initf=bias_initializer,
                                      shape = bshape)
            self.bias_add = biasadd(dtype = None,
                                    init_shape= bshape)

    def apply_biasadd(self, x):
        assert(self.bias_shape[0] < 0 or self.bias_shape == (x.shape[-1],))
        self.bias_shape = (x.shape[-1],)
        if self.use_bias:
            self.bias_weight.initialize(x.elt_type,
                                        self.bias_shape)
            y = self.bias_add(x, self.bias_weight.res)
            return y
        else:
            return x
