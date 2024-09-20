from keraslus.utilities.op_base import TfOp

class InputLayer(TfOp):
    def __init__(
            self,
            input_shape=None,
            batch_size=None,
            dtype=None,
            input_tensor=None,
            sparse=None,
            name=None,
            ragged=None,
            type_spec=None,
            **kwargs):
        # assert(input_tensor != None)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.mlir_name = "\"tf.Identity\""
        self.dtype = dtype
        self.sparse = sparse
        self.input_tensor = input_tensor
        self.ragged = ragged
        self.type_spec = type_spec
        super().__init__()
        # super().__init__(dtype=dtype, init_shape=(batch_size,) + input_shape)

    def __call__(self, x):
        self.args = (x,)
        self.res.shape = x.shape
        return self.res

    # def __str__(self):
    #     return (str(self.res) + " = InputLayer " + str(self.args[0]) + " -> " + str(self.res.shape))
