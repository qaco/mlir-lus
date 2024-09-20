#!/bin/bash

# mlir/lus to llvmir
mlirlus lstm.lus.mlir --all-fbys-on-base-clock --fbys-centralization \
	--explicit-signals --recompute-order --explicit-clocks \
	--scf-clocks --node-to-reactive-func --sync-to-std | \
    tf-opt --xla-legalize-tf=allow-partial-conversion \
	   --hlo-legalize-to-linalg --canonicalize | \
    mlir-opt --linalg-bufferize \
	     --convert-linalg-to-affine-loops --lower-affine \
	     --scf-bufferize --std-bufferize \
	     --tensor-constant-bufferize --tensor-bufferize --canonicalize \
	     --convert-scf-to-std  --canonicalize \
    	     --test-math-polynomial-approximation \
    	     --std-expand \
	     --convert-math-to-llvm --convert-arith-to-llvm \
	     --convert-std-to-llvm \
	     --convert-memref-to-llvm --canonicalize
    	     # --convert-memref-to-llvm \
    	     # --convert-math-to-llvm --convert-arith-to-llvm \
    	     # --convert-std-to-llvm --canonicalize \
    # mlir-translate --mlir-to-llvmir > lstm.lus.bc
