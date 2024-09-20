from keraslus.utilities.op_base import TfOp
from keraslus.utilities import lit

class Activation(TfOp):
    def __init__(self, activation, **kwargs):
        if activation == lit.ACT_RELU:
            self.mlir_name = "\"tf.Relu\""
        elif activation == lit.ACT_SOFTMAX:
            self.mlir_name = "\"tf.Softmax\""
        elif activation == lit.ACT_LINEAR:
            self.mlir_name = "\"tf.Identity\""
        elif activation == lit.ACT_SIGMOID:
            self.mlir_name = "\"tf.Sigmoid\""
        elif activation == "tanh":
            self.mlir_name = "\"tf.Tanh\""
        else:
            assert(False)
        super().__init__()

    def __call__(self, x):
        self.args = (x,)
        self.res.shape = x.shape
        return self.res

class WithActivation:
    def __init__(self,
                 activation):
        if activation != None:
            self.activation_function = Activation(activation)
        else:
            self.activation_function = None

    def activate(self, x):
        if self.activation_function != None :
            return self.activation_function(x)
        else:
            return x
