from keraslus.utilities.op_base import Op
from keraslus.utilities import lit

class ExtractOp(Op):
    def __init__(self, dtype=None, init_shape=lit.POLY_SHAPE_XD):
        super().__init__(dtype, init_shape)
        self.mlir_name = "tensor.extract"

    def __call__(self, x, indices=()):
        self.indices = indices
        self.tensor = x
        self.args = (x,) + indices
        self.res.shape = x.shape
        self.res.elt_type = x.elt_type
        return self.res

    def __str__(self):
        myOp = self.res_string() + " = " + self.mlir_name + " "
        myTensor = self.tensor.mlir_name
        myIndices = "["
        if len(self.indices) > 0:
            myIndices += self.indices[0].mlir_name
        for ind in self.indices[1:]:
            myIndices += ", " + ind.mlir_name
        myIndices += "]"
        mySig = " : " + self.sig_string()
        return myOp + myTensor + myIndices + mySig
