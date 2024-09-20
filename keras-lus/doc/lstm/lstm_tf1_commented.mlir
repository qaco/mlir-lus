module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 904 : i32}}  {
  func @main(%arg0: tensor<3x1xf32>) -> tensor<?x50xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input_1", outputs = "lstm/strided_slice_3"}} {
    %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<3x50xf32>} : () -> tensor<3x50xf32>
    %cst_0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %cst_1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %cst_2 = "tf.Const"() {value = dense<[1, 0, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
    %cst_3 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
    %cst_4 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %cst_5 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
    %cst_6 = "tf.Const"() {value = dense<[-1, 50]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_7 = "tf.Const"() {value = dense<[-1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %0 = "tf.VarHandleOp"() {_class = ["loc:@lstm/lstm_cell/bias"], allowed_devices = [], container = "", device = "", shared_name = "lstm/lstm_cell/bias"} : () -> tensor<!tf_type.resource<tensor<200xf32>>>
    %1 = "tf.VarHandleOp"() {_class = ["loc:@lstm/lstm_cell/kernel"], allowed_devices = [], container = "", device = "", shared_name = "lstm/lstm_cell/kernel"} : () -> tensor<!tf_type.resource<tensor<1x200xf32>>>
    %2 = "tf.VarHandleOp"() {_class = ["loc:@lstm/lstm_cell/recurrent_kernel"], allowed_devices = [], container = "", device = "", shared_name = "lstm/lstm_cell/recurrent_kernel"} : () -> tensor<!tf_type.resource<tensor<50x200xf32>>>
    %3 = "tf.Transpose"(%arg0, %cst_2) {device = ""} : (tensor<3x1xf32>, tensor<3xi32>) -> tensor<*xf32>
    %4 = "tf.Shape"(%3) {device = ""} : (tensor<*xf32>) -> tensor<?xi32>
    %5 = "tf.StridedSlice"(%4, %cst_4, %cst_3, %cst_3) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    %6 = "tf.TensorListReserve"(%cst_6, %5) {device = ""} : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x50xf32>>>
    %7 = "tf.TensorListFromTensor"(%3, %cst_7) {device = ""} : (tensor<*xf32>, tensor<2xi32>) -> tensor<!tf_type.variant<tensor<?x1xf32>>>
    %8:11 = "tf.While"(%cst_1, %cst_0, %cst_1, %6, %cst, %cst, %5, %7, %1, %0, %2) {_lower_using_switch_merge = true, _num_original_outputs = 11 : i64, _read_only_resource_inputs = [8, 9, 10], _stateful_parallelism = false, body = @lstm_while_body_1580, cond = @lstm_while_cond_1570, device = "", is_stateless = false, parallel_iterations = 32 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<?x50xf32>>>, tensor<3x50xf32>, tensor<3x50xf32>, tensor<i32>, tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<!tf_type.resource<tensor<1x200xf32>>>, tensor<!tf_type.resource<tensor<200xf32>>>, tensor<!tf_type.resource<tensor<50x200xf32>>>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<?x50xf32>>>, tensor<?x50xf32>, tensor<?x50xf32>, tensor<i32>, tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<!tf_type.resource<tensor<1x200xf32>>>, tensor<!tf_type.resource<tensor<200xf32>>>, tensor<!tf_type.resource<tensor<50x200xf32>>>)
    %9 = "tf.Identity"(%8#3) {device = ""} : (tensor<!tf_type.variant<tensor<?x50xf32>>>) -> tensor<!tf_type.variant<tensor<?x50xf32>>>
    %10 = "tf.TensorListStack"(%9, %cst_6) {device = "", num_elements = -1 : i64} : (tensor<!tf_type.variant<tensor<?x50xf32>>>, tensor<2xi32>) -> tensor<?x?x50xf32>
    %11 = "tf.StridedSlice"(%10, %cst_5, %cst_4, %cst_3) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?x?x50xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?x50xf32>
    return %11 : tensor<?x50xf32>
  }
  func private @lstm_while_body_1580(%arg0: tensor<i32>,
       	       			     %arg1: tensor<i32>,
				     %time: tensor<i32>,
				     %results: tensor<!tf_type.variant<tensor<?x50xf32>>>,
				     %state0: tensor<?x50xf32>,
				     %state1: tensor<?x50xf32>,
				     %total_time: tensor<i32>,
				     %inputs_list: tensor<!tf_type.variant<tensor<?x1xf32>>>,
				     %kernel_ptr: tensor<!tf_type.resource<tensor<1x200xf32>>>,
				     %bias_ptr: tensor<!tf_type.resource<tensor<200xf32>>>,
				     %rec_kernel_ptr: tensor<!tf_type.resource<tensor<50x200xf32>>>)
				     -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<?x50xf32>>>, tensor<?x50xf32>, tensor<?x50xf32>, tensor<i32>, tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<!tf_type.resource<tensor<1x200xf32>>>, tensor<!tf_type.resource<tensor<200xf32>>>, tensor<!tf_type.resource<tensor<50x200xf32>>>) attributes {tf.signature.is_stateful} {
    %cst = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_0 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_1 = "tf.Const"() {value = dense<[0, 150]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_2 = "tf.Const"() {value = dense<[0, 100]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_3 = "tf.Const"() {value = dense<[0, 50]> : tensor<2xi32>} : () -> tensor<2xi32>
    %i0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %i1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %f0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %f1 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %cst_8 = "tf.Const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
    %cst_9 = "tf.Const"() {value = dense<2.000000e-01> : tensor<f32>} : () -> tensor<f32>
    %cst_10 = "tf.Const"() {value = dense<[-1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>

    // useless
    %12 = "tf.AddV2"(%arg0, %i1) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>

    // input
    %input = "tf.TensorListGetItem"(%inputs_list, %time, %cst_10) {device = ""} : (tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<i32>, tensor<2xi32>) -> tensor<?x1xf32>

    // time update
    %new_time = "tf.AddV2"(%time, %i1) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>

    // Fetch weights
    %rec_kernel = "tf.ReadVariableOp"(%rec_kernel_ptr) {device = ""} : (tensor<!tf_type.resource<tensor<50x200xf32>>>) -> tensor<50x200xf32>
    %rec_kernel1 = "tf.StridedSlice"(%rec_kernel, %cst_0, %cst_3, %cst) {begin_mask = 3 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<50x200xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<50x50xf32>
    // StridedSlice(rec_kernel, begin=[0,50], end=[0,100], 1)
    %rec_kernel2 = "tf.StridedSlice"(%rec_kernel, %cst_3, %cst_2, %cst) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<50x200xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<50x50xf32>
    // StridedSlice(rec_kernel, begin=[0,100], end=[0,150], 1)
    %rec_kernel3 = "tf.StridedSlice"(%rec_kernel, %cst_2, %cst_1, %cst) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<50x200xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<50x50xf32>
    // StridedSlice(rec_kernel, begin=[0,150], end=[0,0], 1)
    %rec_kernel4 = "tf.StridedSlice"(%rec_kernel, %cst_1, %cst_0, %cst) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 3 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<50x200xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<50x50xf32>
    %bias = "tf.ReadVariableOp"(%bias_ptr) {device = ""} : (tensor<!tf_type.resource<tensor<200xf32>>>) -> tensor<200xf32>
    %bias_split:4 = "tf.Split"(%i0, %bias) {device = ""} : (tensor<i32>, tensor<200xf32>) -> (tensor<50xf32>, tensor<50xf32>, tensor<50xf32>, tensor<50xf32>)
    %kernel = "tf.ReadVariableOp"(%kernel_ptr) {device = ""} : (tensor<!tf_type.resource<tensor<1x200xf32>>>) -> tensor<1x200xf32>
    %kernel_split:4 = "tf.Split"(%i1, %kernel) {device = ""} : (tensor<i32>, tensor<1x200xf32>) -> (tensor<1x50xf32>, tensor<1x50xf32>, tensor<1x50xf32>, tensor<1x50xf32>)

    %14 = "tf.MatMul"(%state0, %rec_kernel1) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x50xf32>, tensor<50x50xf32>) -> tensor<?x50xf32>
    %15 = "tf.MatMul"(%state0, %rec_kernel2) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x50xf32>, tensor<50x50xf32>) -> tensor<?x50xf32>
    %16 = "tf.MatMul"(%state0, %rec_kernel3) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x50xf32>, tensor<50x50xf32>) -> tensor<?x50xf32>
    %17 = "tf.MatMul"(%state0, %rec_kernel4) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x50xf32>, tensor<50x50xf32>) -> tensor<?x50xf32>
    
    %19 = "tf.MatMul"(%input, %kernel_split#0) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x1xf32>, tensor<1x50xf32>) -> tensor<?x50xf32>
    %20 = "tf.BiasAdd"(%19, %bias_split#0) {data_format = "NHWC", device = ""} : (tensor<?x50xf32>, tensor<50xf32>) -> tensor<?x50xf32>
    %21 = "tf.AddV2"(%20, %14) {device = ""} : (tensor<?x50xf32>, tensor<?x50xf32>) -> tensor<?x50xf32>
    // Recurrent activation : hard sigmoid
    %22 = "tf.Mul"(%21, %cst_9) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    %23 = "tf.AddV2"(%22, %cst_8) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    %24 = "tf.Minimum"(%23, %f1) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    %part1 = "tf.Maximum"(%24, %f0) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    
    %26 = "tf.MatMul"(%input, %kernel_split#1) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x1xf32>, tensor<1x50xf32>) -> tensor<?x50xf32>
    %27 = "tf.BiasAdd"(%26, %bias_split#1) {data_format = "NHWC", device = ""} : (tensor<?x50xf32>, tensor<50xf32>) -> tensor<?x50xf32>
    %28 = "tf.AddV2"(%27, %15) {device = ""} : (tensor<?x50xf32>, tensor<?x50xf32>) -> tensor<?x50xf32>
    // Recurrent activation : hard sigmoid
    %29 = "tf.Mul"(%28, %cst_9) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    %30 = "tf.AddV2"(%29, %cst_8) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    %31 = "tf.Minimum"(%30, %f1) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    %part2 = "tf.Maximum"(%31, %f0) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    
    %33 = "tf.Mul"(%part2, %state1) {device = ""} : (tensor<?x50xf32>, tensor<?x50xf32>) -> tensor<?x50xf32>
    
    %34 = "tf.MatMul"(%input, %kernel_split#2) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x1xf32>, tensor<1x50xf32>) -> tensor<?x50xf32>
    %35 = "tf.BiasAdd"(%34, %bias_split#2) {data_format = "NHWC", device = ""} : (tensor<?x50xf32>, tensor<50xf32>) -> tensor<?x50xf32>
    %36 = "tf.AddV2"(%35, %16) {device = ""} : (tensor<?x50xf32>, tensor<?x50xf32>) -> tensor<?x50xf32>
    // Activation : Relu
    %37 = "tf.Relu"(%36) {device = ""} : (tensor<?x50xf32>) -> tensor<?x50xf32>
    %38 = "tf.Mul"(%part1, %37) {device = ""} : (tensor<?x50xf32>, tensor<?x50xf32>) -> tensor<?x50xf32>
    %new_state_1 = "tf.AddV2"(%33, %38) {device = ""} : (tensor<?x50xf32>, tensor<?x50xf32>) -> tensor<?x50xf32>
    // Activation : Relu
    %part3 = "tf.Relu"(%new_state_1) {device = ""} : (tensor<?x50xf32>) -> tensor<?x50xf32>
    
    %41 = "tf.MatMul"(%input, %kernel_split#3) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x1xf32>, tensor<1x50xf32>) -> tensor<?x50xf32>
    %42 = "tf.BiasAdd"(%41, %bias_split#3) {data_format = "NHWC", device = ""} : (tensor<?x50xf32>, tensor<50xf32>) -> tensor<?x50xf32>
    %43 = "tf.AddV2"(%42, %17) {device = ""} : (tensor<?x50xf32>, tensor<?x50xf32>) -> tensor<?x50xf32>
    // Recurrent activation : hard sigmoid
    %44 = "tf.Mul"(%43, %cst_9) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    %45 = "tf.AddV2"(%44, %cst_8) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    %46 = "tf.Minimum"(%45, %f1) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    %part4 = "tf.Maximum"(%46, %f0) {device = ""} : (tensor<?x50xf32>, tensor<f32>) -> tensor<?x50xf32>
    
    %new_state_0 = "tf.Mul"(%part4, %part3) {device = ""} : (tensor<?x50xf32>, tensor<?x50xf32>) -> tensor<?x50xf32>

    // Save new_state_0 as an output
    %outputs_list = "tf.TensorListSetItem"(%results, %time, %new_state_0) {device = ""} : (tensor<!tf_type.variant<tensor<?x50xf32>>>, tensor<i32>, tensor<?x50xf32>) -> tensor<!tf_type.variant<tensor<?x50xf32>>>
    
    return %12, %arg1, %new_time, %outputs_list, %new_state_0, %new_state_1, %total_time, %inputs_list, %kernel_ptr, %bias_ptr, %rec_kernel_ptr : tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<?x50xf32>>>, tensor<?x50xf32>, tensor<?x50xf32>, tensor<i32>, tensor<!tf_type.variant<tensor<?x1xf32>>>, tensor<!tf_type.resource<tensor<1x200xf32>>>, tensor<!tf_type.resource<tensor<200xf32>>>, tensor<!tf_type.resource<tensor<50x200xf32>>>
  }


  func private @lstm_while_cond_1570(%arg0: tensor<i32>, %arg1: tensor<i32>, %time: tensor<i32>, %arg3: tensor<!tf_type.variant<tensor<?x50xf32>>>, %arg4: tensor<?x50xf32>, %arg5: tensor<?x50xf32>, %total_time: tensor<i32>, %arg7: tensor<!tf_type.variant<tensor<?x1xf32>>>, %arg8: tensor<!tf_type.resource<tensor<1x200xf32>>>, %arg9: tensor<!tf_type.resource<tensor<200xf32>>>, %arg10: tensor<!tf_type.resource<tensor<50x200xf32>>>) -> tensor<i1> {
    %0 = "tf.Less"(%time, %total_time) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

