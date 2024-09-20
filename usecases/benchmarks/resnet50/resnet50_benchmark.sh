#! /bin/bash

echo "Producing protobuf"

python3 produce_protobuf.py > /dev/null 2> /dev/null

echo "Applying TF1 benchmark_model"

benchmark_model --graph=resnet50.pb \
    --input_layer="input_1" \
    --input_layer_shape="1,224,224,3" \
    --input_layer_type="float" \
    --output_layer="conv5_block3_out/Relu" \
    > /dev/null 2> resnet50_benchmark_model.txt

echo "Applying tf-opt and iree-benchmark-module"

tf-mlir-translate resnet50.pb \
		  --graphdef-to-splatted-mlir \
		  --tf-input-arrays=input_1 \
		  --tf-input-shapes=1,224,224,3 \
		  --tf-output-arrays="conv5_block3_out/Relu" | \
    tf-opt \
	--tf-standard-pipeline \
	--xla-legalize-tf 2> /dev/null | \
    iree-opt \
	--iree-mhlo-to-mhlo-preprocessing \
	--iree-mhlo-input-transformation-pipeline | \
    iree-translate \
	--iree-hal-target-backends=vulkan \
	--iree-mlir-to-vm-bytecode-module \
	--iree-vm-bytecode-module-optimize | \
    iree-benchmark-module \
	--driver=vulkan \
	--entry_function=main \
	--function_input="1x224x224x3xf32" \
	--benchmark_repetitions=10 \
	> resnet50_tf_opt_iree_benchmark.txt 2> /dev/null

# rm resnet50.pb
