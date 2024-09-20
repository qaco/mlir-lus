from keraslus.utilities.op_base import TfOp
from keraslus.simple_ops.arith_ops import WithBias
from keraslus.simple_ops.activation import WithActivation
from keraslus.utilities import lit
from keraslus.utilities.weight import Weight
from keraslus.utilities import aux

class Conv2DSoloBase(TfOp):
    def __init__(self, kernel_size,
                 padding, dilations, strides,
                 data_format,
                 filters=lit.POLY_DIM):
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilations = dilations
        self.strides = strides
        super().__init__(init_shape=(lit.POLY_SHAPE_3D +
                                     (filters,)))
        self.set_attr(lit.ATTR_DILATIONS, repr(self.dilations))
        self.set_attr(lit.ATTR_EXP_PAD, "[]")
        self.set_attr(lit.ATTR_PAD, "\"" + self.padding + "\"")
        self.set_attr(lit.ATTR_STRIDES, repr(self.strides))
        self.set_attr(lit.ATTR_CUDNN, "true")
        self.set_attr(lit.ATTR_FORMAT, "\"" + data_format + "\"")
    
class Conv2DSolo(Conv2DSoloBase):
    def __init__(self, filters, kernel_size,
                 padding, dilations, strides,
                 data_format):
        self.filters = filters
        self.mlir_name = "\"tf.Conv2D\""
        super().__init__(kernel_size, padding, dilations, strides,
                         data_format, filters)

    def __call__(self, x, k):
        b, ih, iw, c = x.shape
        oh, ow = aux.padding_conv2d((ih, iw), self.kernel_size,
                                    self.dilations, self.strides,
                                    self.padding)
        self.res.shape = (b, oh, ow, self.filters)
        self.args = (x, k)
        return self.res

class DepthWhiseConv2DNative(Conv2DSoloBase):
    def __init__(self, kernel_size, padding,
                 depth_multiplier, dilations,
                 strides, data_format):
        self.depth_multiplier = depth_multiplier
        self.mlir_name = "\"tf.DepthwiseConv2dNative\""
        super().__init__(kernel_size, padding, dilations, strides,
                         data_format)

    def __call__(self, x, k):
        b, ih, iw, c = x.shape
        oh, ow = aux.padding_conv2d((ih, iw), self.kernel_size,
                                    self.dilations, self.strides,
                                    self.padding)
        self.res.shape = (b, oh, ow, c * k.shape[3])
        self.args = (x, k)
        return self.res

class Conv2DBase(TfOp, WithBias, WithActivation):
    def __init__(
            self, kernel_shape, filters, 
            activation=None,
            kernel_initializer=lit.INIT_GLOROT,
            use_bias=True,
            bias_initializer='zeros',
            name = None, **kwargs):
        self.kernel_shape = kernel_shape
        self.mlir_name = "\"tf.Identity\""
        TfOp.__init__(self,
                    init_shape=lit.POLY_SHAPE_3D+(filters,))
        WithBias.__init__(self, use_bias, (filters,),
                          bias_initializer,
                          None, None,
                          name)
        WithActivation.__init__(self, activation)
        self.kernel_weight = Weight(value = None,
                                    p_name=name,
                                    k_name="kernel:0",
                                    initf = kernel_initializer,
                                    dtype = None,
                                    shape = self.kernel_shape)

    def __call__(self, x):
        x = aux.to_ndims(x, 4)
        self.kernel_weight.initialize(x.elt_type,
                                      (self.kernel_shape[0],
                                       self.kernel_shape[1],
                                       x.shape[3],
                                       self.kernel_shape[3]))
        y = self.conv2d(x, self.kernel_weight.res)
        self.res.shape = self.conv2d.res.shape
        myres = self.apply_biasadd(y)
        myres = self.activate(myres)
        self.args = (myres,)

        return self.res

class DepthwiseConv2D(Conv2DBase):
    def __init__(
            self,
            kernel_size, strides=(1, 1), padding='VALID',
            depth_multiplier=1,
            data_format=lit.NHWC, dilation_rate=(1, 1), activation=None,
            use_bias=True, depthwise_initializer='glorot_uniform',
            bias_initializer='zeros', depthwise_regularizer=None,
            bias_regularizer=None, activity_regularizer=None,
            depthwise_constraint=None,
            bias_constraint=None,
            name = None,
            **kwargs):
        dilations = aux.to_int_square(dilation_rate)
        strides = aux.to_int_square(strides)
        ksize = aux.to_int_pair(kernel_size)
        kshape = (ksize[0], ksize[1], lit.POLY_DIM, depth_multiplier)
        super().__init__(kshape, lit.POLY_DIM, activation,
                         depthwise_initializer, use_bias, bias_initializer, 
                         name)
        self.depth_multiplier = depth_multiplier
        self.conv2d = DepthWhiseConv2DNative(
            ksize, padding,
            depth_multiplier, dilations,
            strides, data_format)
        
class Conv2D(Conv2DBase):
    def __init__(
            self, filters, kernel_size, strides=(1, 1),
            padding=lit.VALID, data_format=lit.NHWC,
            dilation_rate=(1, 1), groups=1, activation=None,
            use_bias=True,
            kernel_initializer=lit.INIT_GLOROT,
            bias_initializer=lit.INIT_ZEROS,
            kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None,
            name = None, **kwargs
    ):
        dilations = aux.to_int_square(dilation_rate)
        strides = aux.to_int_square(strides)
        ksize = aux.to_int_pair(kernel_size)
        kshape = (ksize[0], ksize[1],
                  lit.POLY_DIM, filters)
        super().__init__(kshape, filters, activation,
                         kernel_initializer, use_bias, bias_initializer, 
                         name)
        self.conv2d = Conv2DSolo(filters, ksize, padding,
                                 dilations, strides,
                                 data_format)
        

