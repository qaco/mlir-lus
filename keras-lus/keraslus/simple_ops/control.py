from keraslus.utilities.op_base import TfOp
from keraslus.utilities import lit

class Select(TfOp):
    def __init__(self, dtype=None, init_shape=lit.POLY_SHAPE_XD):
        super().__init__(dtype, init_shape)
        self.mlir_name = "\"tf.Select\""

    def __call__(self, cond, trueBranch, falseBranch):
        assert(trueBranch.shape == falseBranch.shape)
        assert(trueBranch.elt_type == falseBranch.elt_type)
        assert(cond.elt_type == "i1")
        assert(cond.shape == ())
        self.args = (cond, trueBranch, falseBranch)
        self.res.shape = trueBranch.shape
        self.res.elt_type = trueBranch.elt_type
        return self.res
