from keraslus.utilities.op_base import Identity
from keraslus.simple_ops.arith_ops import WithBias
from keraslus.simple_ops.activation import WithActivation
from keraslus.utilities.weight import Weight
from keraslus.utilities import lit
from keraslus.simple_ops.arith_ops import matmul
from keraslus.utilities import aux

class Dense(Identity, WithBias, WithActivation):
    def __init__(
            self,
            units,
            activation=None,
            use_bias=True,
            kernel_initializer=lit.INIT_GLOROT,
            bias_initializer=lit.INIT_ZEROS,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name = None,
            **kwargs
    ):
        self.units = units
        self.kernel_weight = Weight(value = None,
                                    p_name=name,
                                    k_name='kernel:0',
                                    initf = kernel_initializer,
                                    shape = (lit.POLY_SHAPE_1D
                                             + (self.units,)))
        self.matmul = matmul(dtype = None,
                             shape = (lit.POLY_SHAPE_1D
                                      + (self.units,)))
        Identity.__init__(self, init_shape=lit.POLY_SHAPE_1D
                    + (self.units,))
        WithBias.__init__(self, use_bias, (self.units,),
                          bias_initializer,
                          bias_regularizer, bias_constraint,
                          name)
        WithActivation.__init__(self, activation)

    def __call__(self,x):

        x = aux.to_ndims(x, 2)
            
        self.kernel_weight.initialize(x.elt_type,
                                      (x.shape[1],self.units))
            
        myres = self.matmul(x, self.kernel_weight.res)
        myres = self.apply_biasadd(myres)
        myres = self.activate(myres)
        myres = Identity.__call__(self, myres)
        return myres
