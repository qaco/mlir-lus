#!/bin/bash

mlir-opt complex.tensor.mlir \
	 --convert-complex-to-standard \
		--func-bufferize --buffer-results-to-out-params \
		--linalg-bufferize --scf-bufferize \
		--arith-bufferize --std-bufferize \
		--tensor-constant-bufferize --tensor-bufferize \
		--buffer-deallocation \
		--canonicalize \
		--convert-linalg-to-affine-loops --lower-affine \
		--convert-scf-to-std --test-math-polynomial-approximation \
		 --std-expand \
		--convert-arith-to-llvm --convert-std-to-llvm \
		--convert-math-to-llvm --convert-complex-to-llvm
		
		# --convert-math-to-llvm --convert-arith-to-llvm \
		# --convert-std-to-llvm --convert-memref-to-llvm \
		# --reconcile-unrealized-casts \
	# mlir-translate --mlir-to-llvmir

# mlir/lus to llvmir
# mlirlus $< --all-fbys-on-base-clock --fbys-centralization \
# 	--explicit-signals --recompute-order --explicit-clocks \
# 	--scf-clocks --node-to-reactive-func --sync-to-std | \
#     mlir-opt --func-bufferize --buffer-results-to-out-params \
# 	     --linalg-bufferize \
# 	     --scf-bufferize --arith-bufferize --std-bufferize \
# 	     --tensor-constant-bufferize --tensor-bufferize \
# 	     --buffer-deallocation --canonicalize \
# 	     --convert-linalg-to-affine-loops --lower-affine \
# 	     --convert-scf-to-std   --canonicalize \
#     	     --test-math-polynomial-approximation \
#     	     --std-expand \
# 	     --convert-math-to-llvm --convert-arith-to-llvm \
# 	     --convert-std-to-llvm --convert-memref-to-llvm \
# 	     --reconcile-unrealized-casts | \
#     mlir-translate --mlir-to-llvmir > $@

	     # --convert-math-to-llvm --convert-arith-to-llvm \
	     # --convert-std-to-llvm \
	     # --convert-memref-to-llvm --canonicalize |

