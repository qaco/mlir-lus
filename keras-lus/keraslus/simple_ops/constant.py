from keraslus.utilities.op_base import TfOp
import numpy as np

class Constant(TfOp):
    def __init__(
            self,
            value,
            dtype=None,
            shape=None):
        self.args = ()
        self.value = value
        self.mlir_name = "\"tf.Const\""
        super().__init__(dtype=dtype, init_shape=shape)
        self.refresh_attribute()

    def update(self, value, shape):
        self.value = value
        self.res.shape = shape
        self.refresh_attribute()
    
    def refresh_attribute(self):
        if (isinstance(self.value, np.ndarray)):
            np.set_printoptions(threshold = np.prod(self.value.shape))
            val = np.array2string(self.value, separator=', ')
        else:
            val = repr(self.value)
        self.attrs["value"] = "dense<" + val + "> : " + self.res.shape_str()
