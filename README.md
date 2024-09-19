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
  -DCMAKE_INSTALL_PREFIX=$LLVM_PATH/llvm-lus -DCMAKE_BUILD_TYPE=Release \
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
```
