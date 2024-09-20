#!/bin/bash

# mlir/lus to llvmir
mlirlus lstm.lus.mlir --all-fbys-on-base-clock --fbys-centralization \
	--explicit-signals --recompute-order --explicit-clocks \
	--scf-clocks --node-to-reactive-func --sync-to-std | \
    tf-opt --xla-legalize-tf=allow-partial-conversion \
	   --hlo-legalize-to-linalg --canonicalize | \
    mlir-opt --linalg-bufferize \
	     --scf-bufferize --std-bufferize \
	     --tensor-constant-bufferize --tensor-bufferize \
	     --buffer-deallocation --canonicalize \
	     --convert-linalg-to-affine-loops --lower-affine \
	     --convert-scf-to-std   --canonicalize \
    	     --test-math-polynomial-approximation \
    	     --std-expand \
	     --convert-math-to-llvm --convert-arith-to-llvm \
	     --convert-std-to-llvm \
	     --convert-memref-to-llvm --canonicalize |
    mlir-translate --mlir-to-llvmir > lstm.lus.bc
# llvm to assembly
llc -O3 --fp-contract=fast lstm.lus.bc -o=lstm.lus.s
# assembly to object
clang -O3 -c lstm.lus.s -o lstm.lus.o

# interface with scheduler
clang -O3 -c main.c -o main.o
clang -O3 -c memrefs.c -o memrefs.o
# scheduler
clang -O3 -c ../runtime/scheduler.c -o ../runtime/scheduler.o
clang -O3 -c ../runtime/scheduler_io.c -o ../runtime/scheduler_io.o

#link
clang lstm.lus.o \
      ../runtime/scheduler.o ../runtime/scheduler_io.o \
      main.o memrefs.o -lm -o lstm
