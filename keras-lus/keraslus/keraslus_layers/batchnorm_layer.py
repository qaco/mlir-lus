from keraslus.utilities.op_base import TfOp
from keraslus.utilities import lit
from keraslus.utilities import aux
from keraslus.utilities.weight import Weight
from keraslus.utilities.value import Value

# To improve (initializations)
class BatchNormalization(TfOp):
    def __init__(
            self,
            axis=3,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer=lit.INIT_ZEROS,
            gamma_initializer=lit.INIT_ONES,
            moving_mean_initializer=lit.INIT_ZEROS,
            moving_variance_initializer=lit.INIT_ONES,
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
            name = None,
            **kwargs
    ):
        self.beta_weight = Weight(value = None,
                                  p_name=name,
                                  k_name='beta:0',
                                  initf=beta_initializer,
                                  shape=lit.POLY_SHAPE_1D)
        self.gamma_weight = Weight(value = None,
                                  p_name=name,
                                  k_name='gamma:0',
                                  initf=gamma_initializer,
                                  shape=lit.POLY_SHAPE_1D)
        self.moving_mean_weight = Weight(value = None,
                                  p_name=name,
                                  k_name='moving_mean:0',
                                  initf=moving_mean_initializer,
                                  shape=lit.POLY_SHAPE_1D)
        self.moving_variance_weight = Weight(value = None,
                                  p_name=name,
                                  k_name='moving_variance:0',
                                  initf=moving_variance_initializer,
                                  shape=lit.POLY_SHAPE_1D)
        tail_res = ((lit.FLOAT32, lit.POLY_SHAPE_1D, self),
                    (lit.FLOAT32, lit.POLY_SHAPE_1D, self),
                    (lit.FLOAT32, lit.POLY_SHAPE_1D, self),
                    (lit.FLOAT32, lit.POLY_SHAPE_1D, self),
                    (lit.FLOAT32, lit.POLY_SHAPE_XD, self))
        super().__init__(init_shape = lit.POLY_SHAPE_4D, tail_res = tail_res)
        self.mlir_name = "\"tf.FusedBatchNormV3\""
        self.axis = axis
        self.set_attr(lit.ATTR_FORMAT, "\"" + lit.NHWC + "\"")
        self.set_attr(lit.EPSILON, str(epsilon) + " : " + lit.FLOAT32)
        self.set_attr(lit.AVG_FACTOR, "1.0 : " + lit.FLOAT32)
        self.set_attr(lit.IS_TRAINING, "false")
        

    def __call__(self, x):

        x = aux.to_ndims(x, 4)
        self.res.shape = x.shape
        self.res.set_dtype(x.elt_type)

        self.beta_weight.initialize(x.elt_type, (x.shape[self.axis],))
        self.gamma_weight.initialize(x.elt_type, (x.shape[self.axis],))
        self.moving_mean_weight.initialize(x.elt_type, (x.shape[self.axis],))
        self.moving_variance_weight.initialize(x.elt_type, (x.shape[self.axis],))
        
        self.args = (x,
                     self.beta_weight.res,
                     self.gamma_weight.res,
                     self.moving_mean_weight.res,
                     self.moving_variance_weight.res)
        for r in self.tail_res[0:-1]:
            r.shape = (x.shape[self.axis],)
        return self.res
