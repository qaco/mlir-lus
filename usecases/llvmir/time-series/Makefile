# Set up compilation configuration 
include ../makefile.config

# Application name
APPNAME = rnn
APPNAME_SEQ = $(APPNAME)-seq

TF_FILES = $(APPNAME).tf.mlir

MLIR_MEMREF_FILES = rnn-aux.memref.mlir
ALLMLIR =  $(TF_FILES:.mlir=.tensor.memref.mlir) $(MLIR_MEMREF_FILES)

RUNTIME_DIR = ../runtime

CFILES = $(RUNTIME_DIR)/scheduler.c $(RUNTIME_DIR)/scheduler_io.c memrefs.c main.c

$(APPNAME): $(ALLMLIR:.mlir=.mlir.o) $(CFILES:.c=.c.o)
	$(CC) $^ -o $@ $(LDFLAGS)

%.mlir.o: %.mlir.s
	@echo "Compiling $< to $@"
	@$(CC) -c $< -o $@

%.mlir.s: %.mlir.bc
	@echo "Compiling $< to $@"
	@$(LLC) -O3 $< -o=$@

%.mlir.bc: %.llvm.mlir
	@echo "Compiling $< to $@"
	@$(MLIRTRANS) --mlir-to-llvmir $< -o=$@

%.memref.llvm.mlir: %.memref.mlir
	@echo "Compiling $< to $@"
	$(MLIROPT) $< --convert-linalg-to-affine-loops \
		--lower-affine --convert-scf-to-std --canonicalize \
		--convert-complex-to-llvm --convert-memref-to-llvm \
		--convert-math-to-llvm --convert-arith-to-llvm \
		--convert-std-to-llvm --canonicalize -o=$@

%.tensor.memref.mlir: %.tensor.mlir
	@echo "Compiling $< to $@"
	$(MLIROPT) $< --func-bufferize --buffer-results-to-out-params \
		--convert-complex-to-standard --arith-expand --std-expand \
		--tensor-constant-bufferize --scf-bufferize --std-bufferize \
		--arith-bufferize --tensor-bufferize --linalg-bufferize \
		--buffer-deallocation > $@

%.tf.tensor.mlir : %.tf.mlir
	$(TF_OPT) $< -xla-legalize-tf=allow-partial-conversion | \
	$(IREE_OPT) --iree-mhlo-to-mhlo-preprocessing | \
	$(TF_OPT) --shape-to-shape-lowering --convert-shape-to-std \
		--canonicalize | \
	$(IREE_OPT) --iree-mhlo-to-linalg-on-tensors --canonicalize -o=$@

%.tf.mlir : %.lus.mlir
	@echo "Compiling $< to $@"
	$(MLIRLUS) $< --normalize --to-sync-automata --sync-to-std -o $@



%.c.o: %.c
	$(CC) -c $< -o $@

clean:
	rm -f $(APPNAME) *.o
