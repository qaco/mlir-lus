from keraslus.simple_ops.constant import Constant
from keraslus.utilities import aux
from keraslus.utilities import lit
# import tensorflow as tf
# import numpy as np

class Weight(Constant):
    def __init__(
            self,
            value,
            p_name,
            k_name,
            initf,
            dtype=None,
            shape=None):
        self.p_name = p_name
        self.k_name = k_name
        self.initf = initf
        super().__init__(value=value, dtype=dtype, shape=shape)

    def initialize(self, dtype, shape=None):
        if self.initf == lit.INIT_ZEROS:
            self.value = 0.0
        elif self.initf == lit.INIT_ONES:
            self.value = 1.0
        elif self.initf == lit.INIT_GLOROT:
            # curry = tf.keras.initializers.GlorotUniform()
            # tfvalue = curry(shape=shape)
            # nvalue = tfvalue.numpy()
            # nvalue = np.round(nvalue, decimals = 3)
            # self.value = nvalue.tolist()
            self.value = 0.0
        elif self.initf == lit.INIT_ORTHOGONAL:
            # curry = tf.keras.initializers.Orthogonal()
            # tfvalue = curry(shape=shape)
            # nvalue = tfvalue.numpy()
            # nvalue = np.round(nvalue, decimals = 3)
            # self.value = nvalue.tolist()
            self.value = 0.0
        else:
            assert(False)
        self.res.set_dtype(dtype)
        if shape != None:
            self.res.shape = shape
        self.refresh_attribute()

    def load(self, value):
        self.value = value
        self.refresh_attribute()
        
