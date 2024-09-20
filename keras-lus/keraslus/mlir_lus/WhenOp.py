from keraslus.utilities.op_base import Op
from keraslus.utilities import lit

class WhenOp(Op):
    def __init__(self, dtype=None, init_shape=lit.POLY_SHAPE_XD):
        super().__init__(dtype, init_shape)
        self.mlir_name = "lus.when"

    def __call__(self, cond, data):
        self.cond = cond
        self.data = data
        self.res.shape = data.shape
        self.res.elt_type = data.elt_type
        self.args = (cond, data)
        return self.res

    def __str__(self):
        myOp = self.res_string() + " = " + self.mlir_name + " "
        myArgs = self.cond.mlir_name + " " + self.data.mlir_name
        mySig = " : " + self.sig_string()
        return myOp + myArgs + mySig
