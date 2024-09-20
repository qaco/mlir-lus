# Set up compilation configuration 
include ../makefile.config

# Application name
APPNAME = resnet
APPNAME_SEQ = resnet-seq

TF_FILES = $(APPNAME).tf.mlir
MLIR_LUS_FILES = $(APPNAME).lus.mlir
MLIR_MEMREF_FILES = main.memref.mlir

RUNTIME_DIR = ../runtime

CFILES = $(RUNTIME_DIR)/scheduler.c $(RUNTIME_DIR)/scheduler_io.c memrefs.c main.c
CFILES_SEQ = io.c tick.c

ALLMLIR = $(MLIR_LUS_FILES:.mlir=.tf.tensor.memref.mlir) $(TF_FILES:.mlir=.tensor.memref.mlir)
ALLMLIR_SEQ = main.memref.mlir $(MLIR_LUS_FILES:.mlir=.tf.tensor.memref.mlir) $(TF_FILES:.mlir=.tensor.memref.mlir)

$(APPNAME): $(ALLMLIR:.mlir=.mlir.o) $(CFILES:.c=.c.o)
	$(LD) $^ -o $@ $(LDFLAGS)

$(APPNAME_SEQ): $(ALLMLIR_SEQ:.mlir=.mlir.o) $(CFILES_SEQ:.c=.c.o)
	$(LD) $^ -o $(APPNAME_SEQ) $(LDFLAGS)

%.mlir.o: %.mlir.s
	@echo "Compiling $< to $@"
	@$(CC) -c $< -o $@

%.mlir.s: %.mlir.bc
	@echo "Compiling $< to $@"
	@$(LLC) $< -o=$@

%.mlir.bc: %.llvm.mlir
	@echo "Compiling $< to $@"
	@$(MLIRTRANS) --mlir-to-llvmir $< -o=$@

%.memref.llvm.mlir: %.memref.mlir
	@echo "Compiling $< to $@"
	$(MLIROPT) $< --convert-linalg-to-affine-loops \
		--lower-affine --convert-scf-to-std --std-expand --canonicalize \
		--convert-complex-to-llvm --convert-memref-to-llvm \
		--convert-math-to-llvm --convert-arith-to-llvm \
		--convert-std-to-llvm --canonicalize -o=$@

%.tensor.memref.mlir: %.tensor.mlir
	@echo "Compiling $< to $@"
	$(MLIROPT) $< --func-bufferize --buffer-results-to-out-params \
		--convert-complex-to-standard --arith-expand --std-expand \
		--tensor-constant-bufferize --scf-bufferize --std-bufferize \
		--arith-bufferize --tensor-bufferize --linalg-bufferize \
		--canonicalize | \
	$(MLIRPRIME) --remove-simple-linalg-copy --remove-simple-memref-copy -o=$@

%.lus.tf.mlir: %.lus.mlir
	@echo "Compiling $< to $@"
	$(MLIRLUS) $< --all-fbys-on-base-clock --fbys-centralization \
		--explicit-signals --recompute-order --explicit-clocks \
		--scf-clocks --node-to-reactive-func --sync-to-std -o $@

%.tf.tensor.mlir : %.tf.mlir
	@echo "Compiling $< to $@"
	$(TF_OPT) $< -xla-legalize-tf=allow-partial-conversion | \
	$(IREE_OPT) --iree-mhlo-to-mhlo-preprocessing | \
	$(TF_OPT) --shape-to-shape-lowering --convert-shape-to-std \
		--canonicalize | \
	$(IREE_OPT) --iree-mhlo-to-linalg-on-tensors --canonicalize -o=$@

%.c.o: %.c
	$(CC) -c $< -o $@

clean:
	rm -f $(APPNAME) $(APPNAME_SEQ) *.o
