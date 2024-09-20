from keraslus.utilities.op_base import TfOp

# input_shape=(8,8,5)
# inp = keras.Input(shape=input_shape)
# x = layers.LayerNormalization(
#         axis=[2,3],
#         epsilon=0.001,
#         center=True,
#         scale=True,
#         beta_initializer="zeros",
#         gamma_initializer="ones",
#         beta_regularizer=None,
#         gamma_regularizer=None,
#         beta_constraint=None,
#         gamma_constraint=None)(inp)
# model = keras.Model(inp, x, name="mynet")

# module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 898 : i32}}  {
#   func @main(%arg0: tensor<1x8x8x5xf32>) -> tensor<1x8x8x5xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input_1", outputs = "layer_normalization/add"}} {

#     // The axes of the normalization (2 & 3, ie input[1] & input[2], ie arg0[2] & arg0 [3])
#     %axes = "tf.Const"() {value = dense<[1, 1, 8, 5]> : tensor<4xi32>} : () -> tensor<4xi32>

#     // Gamma according to the axes input[1] & input[2]
#     %0 = "tf.VarHandleOp"() {_class = ["loc:@layer_normalization/beta"], allowed_devices = [], container = "", device = "", shared_name = "layer_normalization/beta"} : () -> tensor<!tf_type.resource<tensor<8x5xf32>>>
#     %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<8x5xf32>>>) -> tensor<8x5xf32>
#     %beta = "tf.Reshape"(%1, %axes) : (tensor<8x5xf32>, tensor<4xi32>) -> tensor<1x1x8x5xf32>

#     // Gamma according to the axes input[1] & input[2]
#     %3 = "tf.VarHandleOp"() {_class = ["loc:@layer_normalization/gamma"], allowed_devices = [], container = "", device = "", shared_name = "layer_normalization/gamma"} : () -> tensor<!tf_type.resource<tensor<8x5xf32>>>
#     %4 = "tf.ReadVariableOp"(%3) : (tensor<!tf_type.resource<tensor<8x5xf32>>>) -> tensor<8x5xf32>
#     %gamma = "tf.Reshape"(%4, %axes) : (tensor<8x5xf32>, tensor<4xi32>) -> tensor<1x1x8x5xf32>

#     %reshape_axes_shape = "tf.Const"() {value = dense<[1, 8, 40, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
#     %6 = "tf.Reshape"(%arg0, %reshape_axes_shape) : (tensor<1x8x8x5xf32>, tensor<4xi32>) -> tensor<1x8x40x1xf32>
#     %scale = "tf.Const"() {value = dense<1.000000e+00> : tensor<8xf32>} : () -> tensor<8xf32>
#     %offset = "tf.Const"() {value = dense<0.000000e+00> : tensor<8xf32>} : () -> tensor<8xf32>
#     %mean = "tf.Const"() {value = dense<> : tensor<0xf32>} : () -> tensor<0xf32>
#     %variance = "tf.Const"() {value = dense<> : tensor<0xf32>} : () -> tensor<0xf32>
#     %y, %batch_mean, %batch_variance, %reserve_space_1, %reserve_space_2, %reserve_space_3 = "tf.FusedBatchNormV3"(%6, %scale, %offset, %mean, %variance) {data_format = "NCHW", epsilon = 1.000000e-03 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<1x8x40x1xf32>, tensor<8xf32>, tensor<8xf32>, tensor<0xf32>, tensor<0xf32>) -> (tensor<1x8x40x1xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>)

#     %shape_input = "tf.Const"() {value = dense<[1, 8, 8, 5]> : tensor<4xi32>} : () -> tensor<4xi32>
#     %7 = "tf.Reshape"(%y, %shape_input) : (tensor<1x8x40x1xf32>, tensor<4xi32>) -> tensor<1x8x8x5xf32>
#     // if scale is true
#     %8 = "tf.Mul"(%7, %gamma) : (tensor<1x8x8x5xf32>, tensor<1x1x8x5xf32>) -> tensor<1x8x8x5xf32>
#     // if center is true
#     %9 = "tf.AddV2"(%8, %beta) : (tensor<1x8x8x5xf32>, tensor<1x1x8x5xf32>) -> tensor<1x8x8x5xf32>
#     return %9 : tensor<1x8x8x5xf32>
#   }
# }

class LayerNormalization(TfOp):
    def __init__(self,
                 axis=-1,
                 epsilon=0.001,
                 center=True,
                 scale=True,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        self.mlir_name = "\"tf.Identity\""
        super().__init__()

    def __call__(self, x):
        self.args = (x,)
        self.res.shape = x.shape
        return self.res
