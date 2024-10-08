lus.node @timeseries(%data: tensor<3x1xf32>) -> (tensor<3x1xf32>) {
  // Building a clock that is true every 5 cycles, starting in the 5th cycle.
  %c0 = arith.constant 0: i32
  %c1 = arith.constant 1: i32
  %c4 = arith.constant 4: i32
  %c5 = arith.constant 5: i32
  %time_cnt = lus.fby %c0 %time_cnt_mod: i32
  %time_cnt_incr = arith.addi %time_cnt, %c1: i32
  %time_cnt_mod = arith.remui %time_cnt_incr, %c5: i32
  %lstm_clk = arith.cmpi "eq", %time_cnt_mod, %c4: i32
  // LSTM organized for streaming (not batch) execution. Recurrence is performed
  // over sections of 5 cycles. Thus, the output is only used when %lstm_clk is true.
  %lstm_out = lus.instance @lstm(%data,%lstm_clk): (tensor<3x1xf32>,i1) -> tensor<3x100xf32>
  // The remaining processing pipeline is only executed when %lstm_clk is true
  %subsampled = lus.when %lstm_clk %lstm_out: tensor<3x100xf32>

  %v0 = arith.constant dense<0.0>: tensor<50xf32>
  %v1 = arith.constant dense<0.0>: tensor<100x50xf32>
  %v2 = arith.constant dense<0.0>: tensor<50xf32>
  %v3 = arith.constant dense<0.0>: tensor<50x50xf32>
  %v4 = arith.constant dense<0.0>: tensor<1xf32>
  %v5 = arith.constant dense<0.0>: tensor<50x1xf32>
  %v21 = "tf.MatMul"(%subsampled, %v1) {device = "", transpose_a = false, transpose_b = false} : (tensor<3x100xf32>, tensor<100x50xf32>) -> tensor<3x50xf32>
  %v22 = "tf.BiasAdd"(%v21, %v0) {data_format = "NHWC", device = ""} : (tensor<3x50xf32>, tensor<50xf32>) -> tensor<3x50xf32>
  %v23 = "tf.Relu"(%v22) {device = ""} : (tensor<3x50xf32>) -> tensor<3x50xf32>
  %v24 = "tf.MatMul"(%v23, %v3) {device = "", transpose_a = false, transpose_b = false} : (tensor<3x50xf32>, tensor<50x50xf32>) -> tensor<3x50xf32>
  %v25 = "tf.BiasAdd"(%v24, %v2) {data_format = "NHWC", device = ""} : (tensor<3x50xf32>, tensor<50xf32>) -> tensor<3x50xf32>
  %v26 = "tf.Relu"(%v25) {device = ""} : (tensor<3x50xf32>) -> tensor<3x50xf32>
  %v27 = "tf.MatMul"(%v26, %v5) {device = "", transpose_a = false, transpose_b = false} : (tensor<3x50xf32>, tensor<50x1xf32>) -> tensor<3x1xf32>
  %res = "tf.BiasAdd"(%v27, %v4) {data_format = "NHWC", device = ""} : (tensor<3x1xf32>, tensor<1xf32>) -> tensor<3x1xf32>
  lus.yield (%res: tensor<3x1xf32>)
}

lus.node @lstm(%data: tensor<3x1xf32>,%rst:i1) -> (tensor<3x100xf32>) {
  %zero_tensor_cst = arith.constant dense<0.0>: tensor<3x100xf32>
  %tmp0 = lus.fby %zero_tensor_cst %state0out: tensor<3x100xf32>
  %tmp1 = lus.fby %zero_tensor_cst %state1out: tensor<3x100xf32>
  %v24 = select %rst, %zero_tensor_cst, %tmp0 : tensor<3x100xf32>
  %v25 = select %rst, %zero_tensor_cst, %tmp1 : tensor<3x100xf32>
  %o76 = arith.constant dense<0.0>: tensor<100x400xf32>
  %o22 = arith.constant dense<0.0>: tensor<1x400xf32>
  %o78 = arith.constant dense<0.0>: tensor<400xf32>
  %v26 = "tf.MatMul"(%v24, %o76) {device = "", transpose_a = false, transpose_b = false} : (tensor<3x100xf32>, tensor<100x400xf32>) -> tensor<3x400xf32>
  %v28 =  "tf.MatMul"(%data, %o22) {device = "", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x400xf32>) -> tensor<3x400xf32>
  %v29 = "tf.AddV2"(%v28, %v26) {device = ""} : (tensor<3x400xf32>, tensor<3x400xf32>) -> tensor<3x400xf32>
  %v30 = "tf.BiasAdd"(%v29, %o78) {data_format = "NHWC", device = ""} : (tensor<3x400xf32>, tensor<400xf32>) -> tensor<3x400xf32>
  //split4
  %split_dim = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %v31_0, %v31_1, %v31_2, %v31_3 = "tf.Split"(%split_dim, %v30): (tensor<i32>, tensor<3x400xf32>) -> (tensor<3x100xf32>, tensor<3x100xf32>, tensor<3x100xf32>, tensor<3x100xf32>)
  //
  %v32 = "tf.Relu"(%v31_2) {device = ""} : (tensor<3x100xf32>) -> tensor<3x100xf32>
  %v33 = "tf.Sigmoid"(%v31_0): (tensor<3x100xf32>) -> tensor<3x100xf32>
  %v34 = "tf.Mul"(%v33, %v32): (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
  %v35 = "tf.Sigmoid"(%v31_1): (tensor<3x100xf32>) -> tensor<3x100xf32>
  %v36 = "tf.Mul"(%v35, %v25): (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
  %state1out = "tf.AddV2"(%v36, %v34) {device = ""} : (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
  %v40 = "tf.Relu"(%state1out) {device = ""} : (tensor<3x100xf32>) -> tensor<3x100xf32>
  %v41 = "tf.Sigmoid"(%v31_3): (tensor<3x100xf32>) -> tensor<3x100xf32>
  %state0out = "tf.Mul"(%v41, %v40): (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>

  lus.yield (%state1out: tensor<3x100xf32>)
}

