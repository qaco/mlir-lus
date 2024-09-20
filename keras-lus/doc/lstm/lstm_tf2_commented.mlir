module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 904 : i32}}  {
  func @__inference_predict_step_func_2826(%arg0: tensor<1x3x1xf32> {tf._user_specified_name = "x"},
       					   %arg1: tensor<1x1xf32> {tf._user_specified_name = "y"},
					   %arg2: tensor<*x!tf_type.resource>, // weight : lstm/lstm_cell/kernel:0
					   %arg3: tensor<*x!tf_type.resource>, // weight : lstm/lstm_cell/recurrent_kernel:0
					   %arg4: tensor<*x!tf_type.resource>, // weight : lstm/lstm_cell/bias:0
					   %arg5: tensor<*x!tf_type.resource>,
					   %arg6: tensor<*x!tf_type.resource>)
					   attributes {tf.entry_function = {control_outputs ="sequential/lstm/lstm_cell/MatMul/ReadVariableOp,
					   	      			   		      sequential/lstm/lstm_cell/MatMul_1/ReadVariableOp,
											      sequential/lstm/lstm_cell/BiasAdd/ReadVariableOp,
											      sequential/lstm/while,
											      sequential/dense/MatMul/ReadVariableOp,sequential/dense/BiasAdd/ReadVariableOp",
					   	      			    inputs ="x,
									    	     y,
										     sequential_lstm_lstm_cell_matmul_readvariableop_resource,sequential_lstm_lstm_cell_matmul_1_readvariableop_resource,
										     sequential_lstm_lstm_cell_biasadd_readvariableop_resource,sequential_dense_matmul_readvariableop_resource,
										     sequential_dense_biasadd_readvariableop_resource",
									    outputs = ""}} {
    %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<1x50xf32>} : () -> tensor<1x50xf32>
    %cst_0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %cst_1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %cst_2 = "tf.Const"() {value = dense<[1, 0, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
    // Number of time steps
    %cst_3 = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
    %cst_4 = "tf.Const"() {value = dense<[1, 50]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_5 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
    // History of outputs
    %0 = "tf.TensorListReserve"(%cst_4, %cst_3) {device = ""} : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<1x50xf32>>>
    %1 = "tf.Transpose"(%arg0, %cst_2) {device = ""} : (tensor<1x3x1xf32>, tensor<3xi32>) -> tensor<3x1x1xf32>
    %2 = "tf.TensorListFromTensor"(%1, %cst_5) {device = ""} : (tensor<3x1x1xf32>, tensor<2xi32>) -> tensor<!tf_type.variant<tensor<1x1xf32>>>
    %3:11 = "tf.While"(%cst_1,
		       %cst_0,
		       %cst_1,
		       %0,
		       %cst,
		       %cst,
		       %cst_3,
		       %2,
		       %arg2,
		       %arg3,
		       %arg4)
		       {_lower_using_switch_merge = true,
		        _num_original_outputs = 11 : i64,
			_read_only_resource_inputs = [8, 9, 10],
			_stateful_parallelism = false,
			body = @sequential_lstm_while_body_27370,
			cond = @sequential_lstm_while_cond_27360,
			device = "", is_stateless = false,
			parallel_iterations = 32 : i64, shape_invariant} : (tensor<i32>,
					      	   			    tensor<i32>,
									    tensor<i32>,
									    tensor<!tf_type.variant<tensor<1x50xf32>>>,
									    tensor<1x50xf32>,
									    tensor<1x50xf32>,
									    tensor<i32>,
									    tensor<!tf_type.variant<tensor<1x1xf32>>>,
									    tensor<*x!tf_type.resource>,
									    tensor<*x!tf_type.resource>,
									    tensor<*x!tf_type.resource>) -> (tensor<i32>,
									    				     tensor<i32>,
													     tensor<i32>,
													     tensor<!tf_type.variant<tensor<1x50xf32>>>,
													     tensor<1x50xf32>,
													     tensor<1x50xf32>,
													     tensor<i32>,
													     tensor<!tf_type.variant<tensor<1x1xf32>>>,
													     tensor<!tf_type.resource>,
													     tensor<!tf_type.resource>,
													     tensor<!tf_type.resource>)
    return
  }
  func private @sequential_lstm_while_body_27370(%arg0: tensor<i32>, // useless
       	       					 %arg1: tensor<i32>, // useless
						 %time: tensor<i32>,
						 %results: tensor<!tf_type.variant<tensor<1x50xf32>>>,
						 %state0: tensor<1x50xf32>, // previous result
						 %state1: tensor<1x50xf32>,
						 %time_steps: tensor<i32>,
						 %inputs_list: tensor<!tf_type.variant<tensor<1x1xf32>>>,
						 %kernel_ptr: tensor<*x!tf_type.resource>, // kernel
						 %rec_kernel_ptr: tensor<*x!tf_type.resource>, // recurrent_kernel
						 %bias_ptr: tensor<*x!tf_type.resource>) // bias
						 -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<1x50xf32>>>, tensor<1x50xf32>, tensor<1x50xf32>, tensor<i32>, tensor<!tf_type.variant<tensor<1x1xf32>>>, tensor<*x!tf_type.resource>, tensor<*x!tf_type.resource>, tensor<*x!tf_type.resource>) attributes {tf._construction_context = "kEagerRuntime", tf.signature.is_stateful} {

    %one = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>

    // useless
    %6 = "tf.AddV2"(%arg0, %one) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = "tf.Identity"(%6) {device = ""} : (tensor<i32>) -> tensor<i32>

    
    %bias = "tf.ReadVariableOp"(%bias_ptr) {device = ""} : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
    %rec_kernel = "tf.ReadVariableOp"(%rec_kernel_ptr) {device = ""} : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
    %kernel = "tf.ReadVariableOp"(%kernel_ptr) {device = ""} : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
    
    %5 = "tf.MatMul"(%state0, %rec_kernel) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x50xf32>, tensor<*xf32>) -> tensor<1x?xf32>
    
    %input_data = "tf.TensorListGetItem"(%inputs_list, %time, %cst_0) {device = ""} : (tensor<!tf_type.variant<tensor<1x1xf32>>>, tensor<i32>, tensor<2xi32>) -> tensor<1x1xf32>
    %10 = "tf.MatMul"(%input_data, %kernel) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x1xf32>, tensor<*xf32>) -> tensor<1x?xf32>
    %11 = "tf.AddV2"(%10, %5) {device = ""} : (tensor<1x?xf32>, tensor<1x?xf32>) -> tensor<1x?xf32>
    %12 = "tf.BiasAdd"(%11, %bias) {data_format = "NHWC", device = ""} : (tensor<1x?xf32>, tensor<*xf32>) -> tensor<1x?xf32>
    %13:4 = "tf.Split"(%one, %12) {device = ""} : (tensor<i32>, tensor<1x?xf32>) -> (tensor<1x?xf32>, tensor<1x?xf32>, tensor<1x?xf32>, tensor<1x?xf32>)
    %14 = "tf.Relu"(%13#2) {device = ""} : (tensor<1x?xf32>) -> tensor<1x?xf32>
    %15 = "tf.Sigmoid"(%13#0) {device = ""} : (tensor<1x?xf32>) -> tensor<1x?xf32>
    %16 = "tf.Mul"(%15, %14) {device = ""} : (tensor<1x?xf32>, tensor<1x?xf32>) -> tensor<1x?xf32>
    %17 = "tf.Sigmoid"(%13#1) {device = ""} : (tensor<1x?xf32>) -> tensor<1x?xf32>
    %18 = "tf.Mul"(%17, %state1) {device = ""} : (tensor<1x?xf32>, tensor<1x50xf32>) -> tensor<1x50xf32>
    %state1out = "tf.AddV2"(%18, %16) {device = ""} : (tensor<1x50xf32>, tensor<1x?xf32>) -> tensor<1x50xf32>
    %21 = "tf.Relu"(%state1out) {device = ""} : (tensor<1x50xf32>) -> tensor<1x50xf32>
    %22 = "tf.Sigmoid"(%13#3) {device = ""} : (tensor<1x?xf32>) -> tensor<1x?xf32>
    %state0out = "tf.Mul"(%22, %21) {device = ""} : (tensor<1x?xf32>, tensor<1x50xf32>) -> tensor<1x50xf32>
    %results_updated = "tf.TensorListSetItem"(%results, %time, %state0out) {device = ""} : (tensor<!tf_type.variant<tensor<1x50xf32>>>, tensor<i32>, tensor<1x50xf32>) -> tensor<!tf_type.variant<tensor<1x50xf32>>>
    
    %next_time = "tf.AddV2"(%time, %one) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    
    return %7, %arg1, %next_time, %results_updated, %state0out, %state1out, %time_steps, %inputs_list, %kernel_ptr, %rec_kernel_ptr, %bias_ptr : tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<1x50xf32>>>, tensor<1x50xf32>, tensor<1x50xf32>, tensor<i32>, tensor<!tf_type.variant<tensor<1x1xf32>>>, tensor<*x!tf_type.resource>, tensor<*x!tf_type.resource>, tensor<*x!tf_type.resource>
  }
  
  func private @sequential_lstm_while_cond_27360(%arg0: tensor<i32>,
       	       					 %arg1: tensor<i32>,
						 %time: tensor<i32>,
						 %arg3: tensor<!tf_type.variant<tensor<1x50xf32>>>,
						 %arg4: tensor<1x50xf32>, %arg5: tensor<1x50xf32>,
						 %time_steps: tensor<i32>,
						 %arg7: tensor<!tf_type.variant<tensor<1x1xf32>>>,
						 %arg8: tensor<*x!tf_type.resource>,
						 %arg9: tensor<*x!tf_type.resource>,
						 %arg10: tensor<*x!tf_type.resource>) -> tensor<i1> attributes {tf._construction_context = "kEagerRuntime"} {
    %0 = "tf.Less"(%time, %time_steps) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
    return %1 : tensor<i1>
  }
}

