# Shape conventions
POLY_DIM = -1
POLY_SHAPE_4D = (POLY_DIM, POLY_DIM, POLY_DIM, POLY_DIM)
POLY_SHAPE_3D = (POLY_DIM, POLY_DIM, POLY_DIM)
POLY_SHAPE_2D = (POLY_DIM, POLY_DIM)
POLY_SHAPE_1D = (POLY_DIM,)
POLY_SHAPE_XD = (-2,)

# Initializers names
INIT_GLOROT = "glorot_uniform"
INIT_ONES = "ones"
INIT_ZEROS = "zeros"
INIT_ORTHOGONAL = "orthogonal"

# Activation functions names
ACT_RELU = "relu"
ACT_SOFTMAX = "softmax"
ACT_LINEAR = "linear"
ACT_SIGMOID = "sigmoid"

# Attributes keys
ATTR_DILATIONS = "dilations"
ATTR_EXP_PAD = "explicit_paddings"
ATTR_PAD = "padding"
ATTR_STRIDES = "strides"
ATTR_CUDNN = "use_cudnn_on_gpu"
ATTR_FORMAT = "data_format"
ATTR_KSIZE = "ksize"
ATTR_KEEP_DIMS = "keep_dims"
ATTR_TRANS_A = "transpose_a"
ATTR_TRANS_B = "transpose_b"
EPSILON = "epsilon"
AVG_FACTOR = "exponential_avg_factor"
IS_TRAINING = "is_training"
BEGIN_MASK = "begin_mask"
END_MASK = "end_mask"
ELLIPSIS_MASK = "ellipsis_mask"
NEW_AXIS_MASK = "new_axis_mask"
SHRINK_AXIS_MASK = "shrink_axis_mask"

# Data formats
NHWC = "NHWC"

# Padding algorithms
VALID = "VALID"
SAME = "SAME"

# Types
FLOAT32 = "f32"
