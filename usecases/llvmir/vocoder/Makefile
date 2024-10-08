# Set up compilation configuration 
include ../makefile.config

APPNAME = pitch

RUNTIME_DIR = ../runtime

CFILES = \
	bitrev.c \
	main.c \
	memrefs.c \
	$(RUNTIME_DIR)/scheduler.c \
	$(RUNTIME_DIR)/scheduler_io.c \

MLIR_LUS_FILES = pitch.lus.mlir

MLIR_TENSOR_FILES = \
	complex.tensor.mlir \
	fft.tensor.mlir \
	pitch.tensor.mlir \

MLIR_MEMREF_FILES = \
	fft.memref.mlir \
	pitch.memref.mlir \
	sndio.memref.mlir \

ALL_MLIR_FILES = $(MLIR_MEMREF_FILES) $(MLIR_TENSOR_FILES:.mlir=.memref.mlir) $(MLIR_LUS_FILES:.mlir=.tensor.memref.mlir)

ALL_OBJ_FILES = $(ALL_MLIR_FILES:.mlir=.mlir.o) $(CFILES:.c=.c.o)

$(APPNAME): $(ALL_OBJ_FILES)
	@echo "Linking..."
	@$(LD) $^ -o $@ $(LDFLAGS)

%.c.o: %.c
	@echo "Compiling $< to $@..."
	@$(CC) -c $< -o $@

%.mlir.o: %.mlir.s
	@echo "Compiling $< to $@..."
	@$(CC) -c $< -o $@

%.mlir.s: %.mlir.bc
	@echo "Compiling $< to $@..."
	@$(LLC) $< -o=$@

%.mlir.bc: %.llvm.mlir
	@echo "Compiling $< to $@..."
	@$(MLIRTRANS) $< --mlir-to-llvmir > $@

%.llvm.mlir: %.std.mlir
	@echo "Compiling $< to $@..."
	@$(MLIROPT) $< --convert-complex-to-llvm --convert-memref-to-llvm \
		--convert-math-to-llvm  --convert-std-to-llvm \
		--canonicalize > $@

%.memref.std.mlir: %.memref.mlir
	@echo "Compiling $< to $@..."
	@$(MLIROPT) $< --convert-linalg-to-affine-loops \
		--lower-affine --convert-scf-to-std > $@

%.tensor.memref.mlir: %.tensor.mlir
	@echo "Compiling $< to $@..."
	@$(MLIROPT) $< --std-bufferize \
		--func-bufferize --buffer-results-to-out-params \
		--tensor-constant-bufferize --scf-bufferize  \
		--tensor-bufferize --canonicalize > $@

%.lus.tensor.memref.mlir.bc: %.lus.mlir
	@echo "Compiling $< to $@..."
	mlirlus $< --all-fbys-on-base-clock --fbys-centralization \
		--explicit-signals --recompute-order --explicit-clocks \
		--scf-clocks --node-to-reactive-func --sync-to-std | \
	mlir-opt \
		--func-bufferize --buffer-results-to-out-params \
		--linalg-bufferize --scf-bufferize \
		--arith-bufferize --std-bufferize \
		--tensor-constant-bufferize --tensor-bufferize \
		--buffer-deallocation \
		--canonicalize \
		--convert-linalg-to-affine-loops --lower-affine \
		--convert-scf-to-std --test-math-polynomial-approximation \
		--std-expand \
		--convert-math-to-llvm --convert-arith-to-llvm \
		--convert-std-to-llvm --convert-memref-to-llvm \
		--reconcile-unrealized-casts | \
	mlir-translate --mlir-to-llvmir > $@

run: $(APPNAME)
	mkfifo kbd
	(rec -q -t raw -r 44100 -e signed -b 16 -c 2 - | \
		nice -20 ./$< | \
		play -q -t raw -r 44100 -e signed -b 16 -c 2 - 2>/dev/null&)
	cat >kbd

kill:
	rm -f kbd
	killall -9 $(APPNAME)

clean:
	rm *.o
