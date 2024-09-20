#! /bin/bash

python source.py

# The values you have to set for tf-input-arrays and tf-output-arrays are
# printed by source.py
tf-mlir-translate protobuf.pb \
		  --graphdef-to-splatted-mlir \
		  --tf-input-arrays=input_1 \
		  --tf-input-shapes=3,1 \
		  --tf-output-arrays="lstm/strided_slice_3" | \
    tf-opt --tf-standard-pipeline > target.mlir

rm protobuf.pb
