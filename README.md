# Intall the MLIR tool chain

## Dependencies

* In Debian repos: ```sudo apt install cmake ninja-build clang lld```
* Bazelisk: https://github.com/bazelbuild/bazelisk/releases

## Main IREE repo

```
cd $IREE_INSTALL_PATH
git clone git@github.com:openxla/iree.git
git checkout 02cfcd1e5561c0db0470d09644a1f527c259c208
git submodule update --init
```

## Install mlir-opt

```
cd $IREE_INSTALL_PATH/iree/third_party/llvm-project
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON \
  -DCMAKE_INSTALL_PREFIX=$LLVM_PATH/llvm -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_ASM_COMPILER=clang -DCMAKE_SHARED_LINKER_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fno-omit-frame-pointer" ../llvm

make -j4
make install
```

## Install tf-opt

```
cd $IREE_INSTALL_PATH/iree/third_party/tensorflow
bazelisk build //tensorflow/compiler/mlir:tf-opt
```

## Install iree

```
cd $IREE_INSTALL_PATH/iree
cmake -GNinja -B ../iree-build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON
```
