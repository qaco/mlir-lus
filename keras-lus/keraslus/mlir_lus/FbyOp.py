from keraslus.utilities.op_base import Op
from keraslus.utilities import lit

class FbyOp(Op):
    def __init__(self,
                 dtype=None,
                 init_shape=lit.POLY_SHAPE_XD):
                 # next_cycle = None):
        super().__init__(dtype, init_shape)
        self.mlir_name = "lus.fby"
        # if next_cycle == None:
        #     self.next_cycle = Value(dtype=dtype, init_shape=init_shape)
        # else:
        #     self.next_cycle = next_cycle

    def __call__(self, init, next_cycle):
        self.init = init
        self.next_cycle = next_cycle
        self.res.shape = init.shape
        self.args = (self.init, self.next_cycle)
        self.res.elt_type = init.elt_type
        return self.res

    def __str__(self):
        myOp = self.res_string() + " = " + self.mlir_name + " "
        myArgs = self.init.mlir_name + " " + self.next_cycle.mlir_name
        mySig = " : " + self.sig_string()
        return myOp + myArgs + mySig
