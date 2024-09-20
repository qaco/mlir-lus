  lus.node @model(%arg0:tensor<1x40xf32>) -> (tensor<1x4xf32>) {
    // Period of the recurrence
    %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %1 = lus.fby %0 %3:tensor<i32>
    %2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tf.AddV2"(%1, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = "tf.Const"() {value = dense<49> : tensor<i32>} : () -> tensor<i32>
    %5 = "tf.FloorMod"(%3, %4) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = "tf.LessEqual"(%5, %4) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %period = tensor.extract %6[] : tensor<i1>
    // LSTM
    %LSTM_out = lus.instance @lstm(%arg0,%period): (tensor<1x40xf32>,i1) -> (tensor<1x4xf32>)
    %LSTM_cond_out = lus.when %period %LSTM_out : tensor<1x4xf32>
    %dense_out = lus.instance @dense0(%LSTM_cond_out): (tensor<1x4xf32>) -> (tensor<1x4xf32>)
    // Output
    lus.yield(%dense_out: tensor<1x4xf32>)
  }

  lus.node @lstm(%sample:tensor<1x40xf32>,%rst: i1) -> (tensor<1x4xf32>) {

    // Recurrent kernel
    %rec_kern = "tf.Const"() {value = dense<0.000000e+00> : tensor<4x16xf32>} : () -> tensor<4x16xf32>
    // Regular kernel
    %kern_raw = "tf.Const"() {value = dense<0.000000e+00> : tensor<40x16xf32>} : () -> tensor<40x16xf32>
    %8 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %kern:4 = "tf.Split"(%8, %kern_raw) : (tensor<i32>, tensor<40x16xf32>) -> (tensor<40x4xf32>, tensor<40x4xf32>, tensor<40x4xf32>, tensor<40x4xf32>)
    // Bias
    %bias_raw = "tf.Const"() {value = dense<0.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %12 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %bias:4 = "tf.Split"(%12, %bias_raw) : (tensor<i32>, tensor<16xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)

    // STM management
    %STM_init = "tf.Const"() {value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
    %STM_base = lus.fby %STM_init %STM_up : tensor<1x4xf32>
    %STM_ok = lus.when %rst %STM_base : tensor<1x4xf32>
    %STM_rst = lus.when not %rst %STM_init : tensor<1x4xf32> 
    %STM = lus.merge %rst %STM_ok %STM_rst : tensor<1x4xf32>
    // LTM management
    %LTM_init = "tf.Const"() {value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
    %LTM_base = lus.fby %LTM_init %LTM_up:tensor<1x4xf32>
    %LTM_ok = lus.when %rst %LTM_base : tensor<1x4xf32>
    %LTM_rst = lus.when not %rst %LTM_init : tensor<1x4xf32>
    %LTM = lus.merge %rst %LTM_ok %LTM_rst : tensor<1x4xf32>

    // Forget gate
    %29 = "tf.Const"() {value = dense<[0, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
    %30 = "tf.Const"() {value = dense<[0, 8]> : tensor<2xi32>} : () -> tensor<2xi32>
    %31 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
    %rec_kern1 = "tf.StridedSlice"(%rec_kern, %29, %30, %31) {begin_mask = 1 : i64, ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4x16xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<4x4xf32>
    // Weighting the STM
    %33 = "tf.MatMul"(%STM, %rec_kern1) {transpose_a = false, transpose_b = false} : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
    // Weighting the input sample
    %27 = "tf.MatMul"(%sample, %kern#1) {transpose_a = false, transpose_b = false} : (tensor<1x40xf32>, tensor<40x4xf32>) -> tensor<1x4xf32>
    %28 = "tf.BiasAdd"(%27, %bias#1) {data_format = "NHWC"} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
    // Weighted sum and activation
    %34 = "tf.AddV2"(%28, %33) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %forget_gate = "tf.Sigmoid"(%34) : (tensor<1x4xf32>) -> tensor<1x4xf32>

    // Input gate
    %42 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
    %43 = "tf.Const"() {value = dense<[0, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
    %44 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
    %rec_kern0 = "tf.StridedSlice"(%rec_kern, %42, %43, %44) {begin_mask = 3 : i64, ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4x16xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<4x4xf32>
    // Weighting the STM
    %46 = "tf.MatMul"(%STM, %rec_kern0) {transpose_a = false, transpose_b = false} : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
    // Weighting the input sample
    %40 = "tf.MatMul"(%sample, %kern#0) {transpose_a = false, transpose_b = false} : (tensor<1x40xf32>, tensor<40x4xf32>) -> tensor<1x4xf32>
    %41 = "tf.BiasAdd"(%40, %bias#0) {data_format = "NHWC"} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
    // Weighted sum and activation
    %47 = "tf.AddV2"(%41, %46) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %input_gate = "tf.Sigmoid"(%47) : (tensor<1x4xf32>) -> tensor<1x4xf32>

    // LTM candidate
    %51 = "tf.Const"() {value = dense<[0, 8]> : tensor<2xi32>} : () -> tensor<2xi32>
    %52 = "tf.Const"() {value = dense<[0, 12]> : tensor<2xi32>} : () -> tensor<2xi32>
    %53 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
    %rec_kern2 = "tf.StridedSlice"(%rec_kern, %51, %52, %53) {begin_mask = 1 : i64, ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4x16xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<4x4xf32>
    // Weighting the STM
    %55 = "tf.MatMul"(%STM, %rec_kern2) {transpose_a = false, transpose_b = false} : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
    // Weighting the input sample
    %49 = "tf.MatMul"(%sample, %kern#2) {transpose_a = false, transpose_b = false} : (tensor<1x40xf32>, tensor<40x4xf32>) -> tensor<1x4xf32>
    %50 = "tf.BiasAdd"(%49, %bias#2) {data_format = "NHWC"} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
    // Weighted sum and activation
    %56 = "tf.AddV2"(%50, %55) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %LTM_candidate = "tf.Tanh"(%56) : (tensor<1x4xf32>) -> tensor<1x4xf32>

    // Output gate
    %20 = "tf.Const"() {value = dense<[0, 12]> : tensor<2xi32>} : () -> tensor<2xi32>
    %21 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
    %22 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
    %rec_kern3 = "tf.StridedSlice"(%rec_kern, %20, %21, %22) {begin_mask = 1 : i64, ellipsis_mask = 0 : i64, end_mask = 3 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4x16xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<4x4xf32>
    // Weighting the STM
    %24 = "tf.MatMul"(%STM, %rec_kern3) {transpose_a = false, transpose_b = false} : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
    // Weighting the input sample
    %11 = "tf.MatMul"(%sample, %kern#3) {transpose_a = false, transpose_b = false} : (tensor<1x40xf32>, tensor<40x4xf32>) -> tensor<1x4xf32>
    %15 = "tf.BiasAdd"(%11, %bias#3) {data_format = "NHWC"} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
    // Weighted sum and activation
    %25 = "tf.AddV2"(%15, %24) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %output_gate = "tf.Sigmoid"(%25) : (tensor<1x4xf32>) -> tensor<1x4xf32>

    // LTM update
    %58 = "tf.Mul"(%input_gate, %LTM_candidate) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %39 = "tf.Mul"(%forget_gate, %LTM) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %LTM_up = "tf.AddV2"(%39, %58) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>

    // STM update
    %STM_candidate = "tf.Tanh"(%LTM_up) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %STM_up = "tf.Mul"(%output_gate, %STM_candidate) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    
    lus.yield  (%STM_up:tensor<1x4xf32>)
  }

  lus.node @dense0(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>) {
    %63 = "tf.Const"() {value = dense<0.000000e+00> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %64 = "tf.MatMul"(%arg0, %63) {transpose_a = false, transpose_b = false} : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
    %65 = "tf.Const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
    %66 = "tf.BiasAdd"(%64, %65) {data_format = "NHWC"} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
    %67 = "tf.Relu"(%66) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    lus.yield(%67: tensor<1x4xf32>)
  }

