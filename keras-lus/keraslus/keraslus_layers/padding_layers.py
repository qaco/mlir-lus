from keraslus.utilities import lit
from keraslus.utilities.op_base import TfOp
from keraslus.simple_ops.constant import Constant
from keraslus.simple_ops.reshape_ops import reshape

class ZeroPadding2D(TfOp):
    def __init__(self, padding=(1, 1), data_format=lit.NHWC, **kwargs):
        if type(padding) == int:
            self.padding = ((padding, padding), (padding, padding))
        elif type(padding) == tuple and type(padding[0]) == int:
            self.padding = ((padding[0], padding[0]),(padding[1],padding[1]))
        elif type(padding) == tuple and type(padding[0]) == tuple:
            self.padding = padding
        else:
            assert(False)
        self.data_format = data_format
        myVal = [[0, 0], [self.padding[0][0], self.padding[0][1]],
                 [self.padding[1][0], self.padding[1][1]], [0, 0]]
        self.gen_dep = Constant(myVal, "tf.int32", (4, 2))
        self.mlir_name = "\"tf.Pad\""
        super().__init__(init_shape=lit.POLY_SHAPE_4D)
        self.set_attr(lit.ATTR_FORMAT, "\"" + data_format + "\"")

    def __call__(self, x):
        if (len(x.shape) == 3):
            x = reshape(tshape=[1, x.shape[0],
                                x.shape[1], x.shape[2]])(x)
        self.args = (x, self.gen_dep.res)
        b, h, w, c = x.shape
        nh = h + self.padding[0][0] + self.padding[0][1]
        nw = w + self.padding[1][0] + self.padding[1][1]
        self.res.shape = (b, nh, nw, c)
        return self.res
