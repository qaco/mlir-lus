// mlirlus ok-lstm.mlir --all-fbys-on-base-clock --fbys-centralization --explicit-signals --recompute-order --explicit-clocks --scf-clocks --node-to-reactive-func --sync-to-std

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
    // Dense
    %dense_out = lus.instance @dense0(%LSTM_out): (tensor<1x4xf32>) -> (tensor<1x4xf32>)
    lus.yield(%dense_out: tensor<1x4xf32>)
  }

  lus.node @lstm(%sample:tensor<1x40xf32>,%rst: i1) -> (%o: tensor<1x4xf32>)
    clock { lus.on_clock_node ((base,base) -> (base on %rst)) }{
    // Weights initializations
    %rec_kern = "tf.Const"() {value = dense<2.0> : tensor<4x16xf32>} : () -> tensor<4x16xf32>
    %kern = "tf.Const"() {value = dense<3.0> : tensor<40x16xf32>} : () -> tensor<40x16xf32>
    %bias = "tf.Const"() {value = dense<4.0> : tensor<16xf32>} : () -> tensor<16xf32>
    %STM_init = "tf.Const"() {value = dense<5.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
    %LTM_init = "tf.Const"() {value = dense<6.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
    // STM (hidden state) management
    %STM_base = lus.fby %STM_init %STM_up : tensor<1x4xf32>
    %STM_ok = lus.when %rst %STM_base : tensor<1x4xf32>
    %STM_rst = lus.when not %rst %STM_init : tensor<1x4xf32> 
    %STM = lus.merge %rst %STM_ok %STM_rst : tensor<1x4xf32>
    // LTM (cell state) management
    %LTM_base = lus.fby %LTM_init %LTM_up:tensor<1x4xf32>
    %LTM_ok = lus.when %rst %LTM_base : tensor<1x4xf32>
    %LTM_rst = lus.when not %rst %LTM_init : tensor<1x4xf32>
    %LTM = lus.merge %rst %LTM_ok %LTM_rst : tensor<1x4xf32>
    // Weighted sums
    %w_STM = "tf.MatMul"(%STM, %rec_kern) : (tensor<1x4xf32>, tensor<4x16xf32>) -> tensor<1x16xf32>
    %w_sample = "tf.MatMul"(%sample, %kern) : (tensor<1x40xf32>, tensor<40x16xf32>) -> tensor<1x16xf32>
    %tmp0 = "tf.AddV2"(%w_STM, %w_sample) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %tmp1 = "tf.BiasAdd"(%tmp0, %bias) : (tensor<1x16xf32>, tensor<16xf32>) -> tensor<1x16xf32>
    %dim_split = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %split:4 = "tf.Split"(%dim_split, %tmp1) : (tensor<i32>, tensor<1x16xf32>) -> (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>)
    // LSTM gates
    %input_gate = "tf.Sigmoid"(%split#0) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %forget_gate = "tf.Sigmoid"(%split#1) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %output_gate = "tf.Sigmoid"(%split#3) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    // LTM/STM updates
    %LTM_candidate = "tf.Tanh"(%split#2) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %tmp2 = "tf.Mul"(%input_gate, %LTM_candidate) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %tmp3 = "tf.Mul"(%forget_gate, %LTM) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %LTM_up = "tf.AddV2"(%tmp3, %tmp2) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %STM_candidate = "tf.Tanh"(%LTM_up) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %STM_up = "tf.Mul"(%output_gate, %STM_candidate) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    // Output
    %output = lus.when %rst %STM_up: tensor<1x4xf32>
    lus.yield  (%output:tensor<1x4xf32>)
  }

  lus.node @dense0(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>) {
    %63 = "tf.Const"() {value = dense<2.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %64 = "tf.MatMul"(%arg0, %63) : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
    %65 = "tf.Const"() {value = dense<3.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %66 = "tf.BiasAdd"(%64, %65) : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
    %67 = "tf.Relu"(%66) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    lus.yield(%67: tensor<1x4xf32>)
  }

