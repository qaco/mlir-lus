from keraslus.utilities.op_base import TfOp
from keraslus.utilities import lit
from keraslus.simple_ops.constant import Constant

class Reshape(TfOp):
    def __init__(self, tshape=lit.POLY_SHAPE_XD, **kwargs):
        super().__init__(init_shape = tshape)
        self.mlir_name = "\"tf.Reshape\""

    def __call__(self, x, s):
        self.args = (x, self.tshape.res)
        return self.res

class reshape(Reshape):
    def __init__(self, tshape, **kwargs):
        super().__init__(tshape=tuple(tshape))
        self.tshape = Constant(value=tshape,
                               dtype="i32",
                               shape=(len(tshape),))

    def update_tshape(self, tshape):
        self.tshape.update(tshape, (len(tshape),))
        self.res.shape = tuple(self.tshape.value)
        
    def __call__(self, x):
        return super().__call__(x, self.tshape.res)

class Flatten(reshape):
    def __init__(self, data_format=None, **kwargs):
        tshape = (1,) + lit.POLY_SHAPE_1D
        super().__init__(tshape = tshape)

    def __call__(self,x):
        s = 1
        for d in x.shape:
            s *= d
        tshape = (1,s)
        self.update_tshape(tshape)
        return super().__call__(x)
