
lus.node @model(%v0: tensor<3x1xf32>) -> (tensor<3x1xf32>) {

%v4 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
%v5 = lus.fby %v4 %v6 : tensor<i32>
%v2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
%v6 = "tf.AddV2"(%v5, %v2) {} : (tensor<i32>, tensor<i32>) -> tensor<i32>
%v3 = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
%v7 = "tf.FloorMod"(%v6, %v3) {} : (tensor<i32>, tensor<i32>) -> tensor<i32>
%v8 = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
%v9 = "tf.LessEqual"(%v7, %v8) {} : (tensor<i32>, tensor<i32>) -> tensor<i1>
%v10 = tensor.extract %v9[] : tensor<i1>
%v17 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
%v11 = "tf.Const"() {value = dense<0.0> : tensor<1x400xf32>} : () -> tensor<1x400xf32>
%v58, %v59, %v60, %v61 = "tf.Split"(%v17, %v11) {} : (tensor<i32>, tensor<1x400xf32>) -> (tensor<1x100xf32>, tensor<1x100xf32>, tensor<1x100xf32>, tensor<1x100xf32>)
%v40 = "tf.MatMul"(%v0, %v61) {transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x100xf32>) -> tensor<3x100xf32>
%v15 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
%v13 = "tf.Const"() {value = dense<0.0> : tensor<400xf32>} : () -> tensor<400xf32>
%v54, %v55, %v56, %v57 = "tf.Split"(%v15, %v13) {} : (tensor<i32>, tensor<400xf32>) -> (tensor<100xf32>, tensor<100xf32>, tensor<100xf32>, tensor<100xf32>)
%v41 = "tf.BiasAdd"(%v40, %v57) {data_format = "NHWC"} : (tensor<3x100xf32>, tensor<100xf32>) -> tensor<3x100xf32>
%v22 = "tf.Const"() {value = dense<0.0> : tensor<3x100xf32>} : () -> tensor<3x100xf32>
%v23 = lus.fby %v22 %v52 : tensor<3x100xf32>
%v24 = "tf.Select"(%v9, %v23, %v22) {} : (tensor<i1>, tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
%v12 = "tf.Const"() {value = dense<0.0> : tensor<100x400xf32>} : () -> tensor<100x400xf32>
%v71 = "tf.Const"() {value = dense<[0, 300]> : tensor<2xi32>} : () -> tensor<2xi32>
%v72 = "tf.Const"() {value = dense<[0, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
%v73 = "tf.Const"() {value = dense<[1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
%v21 = "tf.StridedSlice"(%v12, %v71, %v72, %v73) {begin_mask = 1: i64, ellipsis_mask = 0: i64, new_axis_mask = 0: i64, end_mask = 3: i64, shrink_axis_mask = 0: i64} : (tensor<100x400xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<100x100xf32>
%v31 = "tf.MatMul"(%v24, %v21) {transpose_a = false, transpose_b = false} : (tensor<3x100xf32>, tensor<100x100xf32>) -> tensor<3x100xf32>
%v42 = "tf.AddV2"(%v41, %v31) {} : (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
%v43 = "tf.Sigmoid"(%v42) {} : (tensor<3x100xf32>) -> tensor<3x100xf32>
%v36 = "tf.MatMul"(%v0, %v59) {transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x100xf32>) -> tensor<3x100xf32>
%v37 = "tf.BiasAdd"(%v36, %v55) {data_format = "NHWC"} : (tensor<3x100xf32>, tensor<100xf32>) -> tensor<3x100xf32>
%v65 = "tf.Const"() {value = dense<[0, 100]> : tensor<2xi32>} : () -> tensor<2xi32>
%v66 = "tf.Const"() {value = dense<[0, 200]> : tensor<2xi32>} : () -> tensor<2xi32>
%v67 = "tf.Const"() {value = dense<[1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
%v19 = "tf.StridedSlice"(%v12, %v65, %v66, %v67) {begin_mask = 1: i64, ellipsis_mask = 0: i64, new_axis_mask = 0: i64, end_mask = 1: i64, shrink_axis_mask = 0: i64} : (tensor<100x400xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<100x100xf32>
%v29 = "tf.MatMul"(%v24, %v19) {transpose_a = false, transpose_b = false} : (tensor<3x100xf32>, tensor<100x100xf32>) -> tensor<3x100xf32>
%v38 = "tf.AddV2"(%v37, %v29) {} : (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
%v39 = "tf.Sigmoid"(%v38) {} : (tensor<3x100xf32>) -> tensor<3x100xf32>
%v25 = "tf.Const"() {value = dense<0.0> : tensor<3x100xf32>} : () -> tensor<3x100xf32>
%v26 = lus.fby %v25 %v50 : tensor<3x100xf32>
%v27 = "tf.Select"(%v9, %v26, %v25) {} : (tensor<i1>, tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
%v49 = "tf.Mul"(%v39, %v27) {} : (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
%v32 = "tf.MatMul"(%v0, %v58) {transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x100xf32>) -> tensor<3x100xf32>
%v33 = "tf.BiasAdd"(%v32, %v54) {data_format = "NHWC"} : (tensor<3x100xf32>, tensor<100xf32>) -> tensor<3x100xf32>
%v62 = "tf.Const"() {value = dense<[0, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
%v63 = "tf.Const"() {value = dense<[0, 100]> : tensor<2xi32>} : () -> tensor<2xi32>
%v64 = "tf.Const"() {value = dense<[1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
%v18 = "tf.StridedSlice"(%v12, %v62, %v63, %v64) {begin_mask = 3: i64, ellipsis_mask = 0: i64, new_axis_mask = 0: i64, end_mask = 1: i64, shrink_axis_mask = 0: i64} : (tensor<100x400xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<100x100xf32>
%v28 = "tf.MatMul"(%v24, %v18) {transpose_a = false, transpose_b = false} : (tensor<3x100xf32>, tensor<100x100xf32>) -> tensor<3x100xf32>
%v34 = "tf.AddV2"(%v33, %v28) {} : (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
%v35 = "tf.Sigmoid"(%v34) {} : (tensor<3x100xf32>) -> tensor<3x100xf32>
%v44 = "tf.MatMul"(%v0, %v60) {transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x100xf32>) -> tensor<3x100xf32>
%v45 = "tf.BiasAdd"(%v44, %v56) {data_format = "NHWC"} : (tensor<3x100xf32>, tensor<100xf32>) -> tensor<3x100xf32>
%v68 = "tf.Const"() {value = dense<[0, 200]> : tensor<2xi32>} : () -> tensor<2xi32>
%v69 = "tf.Const"() {value = dense<[0, 300]> : tensor<2xi32>} : () -> tensor<2xi32>
%v70 = "tf.Const"() {value = dense<[1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
%v20 = "tf.StridedSlice"(%v12, %v68, %v69, %v70) {begin_mask = 1: i64, ellipsis_mask = 0: i64, new_axis_mask = 0: i64, end_mask = 1: i64, shrink_axis_mask = 0: i64} : (tensor<100x400xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<100x100xf32>
%v30 = "tf.MatMul"(%v24, %v20) {transpose_a = false, transpose_b = false} : (tensor<3x100xf32>, tensor<100x100xf32>) -> tensor<3x100xf32>
%v46 = "tf.AddV2"(%v45, %v30) {} : (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
%v47 = "tf.Relu"(%v46) {} : (tensor<3x100xf32>) -> tensor<3x100xf32>
%v48 = "tf.Mul"(%v35, %v47) {} : (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
%v50 = "tf.AddV2"(%v49, %v48) {} : (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
%v51 = "tf.Relu"(%v50) {} : (tensor<3x100xf32>) -> tensor<3x100xf32>
%v52 = "tf.Mul"(%v43, %v51) {} : (tensor<3x100xf32>, tensor<3x100xf32>) -> tensor<3x100xf32>
%v53 = lus.when %v10 %v52 : tensor<3x100xf32>
%v74 = "tf.Const"() {value = dense<0.0> : tensor<100x50xf32>} : () -> tensor<100x50xf32>
%v75 = "tf.MatMul"(%v53, %v74) {transpose_a = false, transpose_b = false} : (tensor<3x100xf32>, tensor<100x50xf32>) -> tensor<3x50xf32>
%v77 = "tf.Const"() {value = dense<0.0> : tensor<50xf32>} : () -> tensor<50xf32>
%v78 = "tf.BiasAdd"(%v75, %v77) {data_format = "NHWC"} : (tensor<3x50xf32>, tensor<50xf32>) -> tensor<3x50xf32>
%v79 = "tf.Relu"(%v78) {} : (tensor<3x50xf32>) -> tensor<3x50xf32>
%v80 = "tf.Const"() {value = dense<0.0> : tensor<50x50xf32>} : () -> tensor<50x50xf32>
%v81 = "tf.MatMul"(%v79, %v80) {transpose_a = false, transpose_b = false} : (tensor<3x50xf32>, tensor<50x50xf32>) -> tensor<3x50xf32>
%v83 = "tf.Const"() {value = dense<0.0> : tensor<50xf32>} : () -> tensor<50xf32>
%v84 = "tf.BiasAdd"(%v81, %v83) {data_format = "NHWC"} : (tensor<3x50xf32>, tensor<50xf32>) -> tensor<3x50xf32>
%v85 = "tf.Relu"(%v84) {} : (tensor<3x50xf32>) -> tensor<3x50xf32>
%v86 = "tf.Const"() {value = dense<0.0> : tensor<50x1xf32>} : () -> tensor<50x1xf32>
%v87 = "tf.MatMul"(%v85, %v86) {transpose_a = false, transpose_b = false} : (tensor<3x50xf32>, tensor<50x1xf32>) -> tensor<3x1xf32>
%v89 = "tf.Const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
%v90 = "tf.BiasAdd"(%v87, %v89) {data_format = "NHWC"} : (tensor<3x1xf32>, tensor<1xf32>) -> tensor<3x1xf32>
lus.yield(%v90: tensor<3x1xf32>)
}

