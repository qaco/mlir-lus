from keraslus.utilities.op_base import Identity

class Dropout(Identity):
    def __init__(self, dtype=None, init_shape=None, **kwargs):
        super().__init__(dtype, init_shape)

    # def __call__(self, x):
        # self.args = (x,)
        # self.res.shape = x.shape
        # return self.res
