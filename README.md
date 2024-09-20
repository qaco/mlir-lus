The ```mlir-lus``` compiler was designed and developed by Hugo Pompougnac (me)
and Dumitru Potop-Butucaru (my PhD supervisor). It provides two MLIR dialects
-- lus and sync -- for specifying and compiling synchronous programs.
The lus dialect, in particular, implements the synchronous dataflow kernel
of Lustre.

See also:
+ My PhD thesis (in french), [Spécification et compilation de réseaux de neurones embarqués](https://theses.hal.science/tel-03997036/file/POMPOUGNAC_Hugo_these_2022.pdf),
supervised by Dumitru Poptop-Butucaru.
+ The article published in TACO and co-written with Ulysse Beaugnon,
  Albert Cohen and Dumitru Potop-Butucaru, [Weaving Synchronous Reactions into the Fabric of SSA-form Compilers](https://dl.acm.org/doi/full/10.1145/3506706)

# Intall the MLIR tool chain

## Dependencies

* In Debian repos: ```sudo apt install cmake ninja-build clang lld```
* Bazelisk: https://github.com/bazelbuild/bazelisk/releases

## Main IREE repo

```
cd mlir-lus
git submodule update --init
cd iree
git submodule update --init
```

## Install mlir-opt

```
cd iree/third_party/llvm-project
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON \
  -DCMAKE_INSTALL_PREFIX=~/llvm-lus -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_ASM_COMPILER=clang -DCMAKE_SHARED_LINKER_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fno-omit-frame-pointer" ../llvm

make -j4
make install
```

## Install tf-opt

```
cd iree/third_party/tensorflow
bazelisk build //tensorflow/compiler/mlir:tf-opt
```

## Install iree

```
cd iree
cmake -GNinja -B ../iree-build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON
cd ../iree-build
ninja
```

# Install mlir-lus

```
cd mlir-lus
make -j4
```

Try the compiler: ```src/mlirlus tests/ok-lstm.mlir --all-fbys-on-base-clock --fbys-centralization --explicit-signals --recompute-order --explicit-clocks --scf-clocks --node-to-reactive-func --sync-to-std```

Get compiler options: ```mlirlus --help```

Read examples: see the directory ```tests```
