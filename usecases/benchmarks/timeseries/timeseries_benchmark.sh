#! /bin/bash

echo "Producing protobuf"

python3 produce_protobuf.py > /dev/null 2> /dev/null

# echo "Applying TF1 benchmark_model"

# benchmark_model --graph=protobuf.pb \
#     --input_layer="input_1" \
#     --input_layer_shape="3,5,1" \
#     --input_layer_type="float" \
#     --output_layer='dense_2/BiasAdd:0' \
#     > /dev/null 2> timeseries_benchmark_model.txt

echo "Applying tf-opt and iree-benchmark-module"

tf-mlir-translate protobuf.pb --graphdef-to-splatted-mlir \
		  --tf-input-arrays=input_1 --tf-input-shapes=3,5,1 \
		  --tf-output-arrays="dense_2/BiasAdd" | \
    tf-opt --tf-standard-pipeline --tf-promote-resources-to-args \
	   --tf-tensor-list-ops-decomposition \
	   -xla-legalize-tf=allow-partial-conversion \
	   -tf-functional-control-flow-to-cfg --canonicalize 2> /dev/null |
    iree-opt \
	--iree-mhlo-to-mhlo-preprocessing \
	--iree-mhlo-input-transformation-pipeline | \
    iree-translate \
	--iree-hal-target-backends=dylib-llvm-aot \
	--iree-mlir-to-vm-bytecode-module \
	--iree-vm-bytecode-module-optimize | \
    iree-benchmark-module \
	--driver=dylib \
	--entry_function=main \
	--function_input="3x5x1xf32" \
	--function_input="50xf32" \
	--function_input="100x50xf32" \
	--function_input="50xf32" \
	--function_input="50x50xf32" \
	--function_input="1xf32" \
	--function_input="50x1xf32" \
	--function_input="400xf32" \
	--function_input="1x400xf32" \
	--function_input="100x400xf32" \
	--benchmark_repetitions=10 \
	> timeseries_tf_opt_iree_benchmark.txt 2> /dev/null

# rm timeseries.pb
