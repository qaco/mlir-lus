// -*- C++ -*- //

#ifndef HELPERS_H
#define HELPERS_H

#include "llvm/ADT/SmallSet.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
  struct Helpers {

    static void feed_with_deps_of(Value v,
				  llvm::SmallSet<Operation*,4> &deps);

    static bool depends_on_inputs(Value v);

    static TensorType abstract_tensor(Type t);

    static MemRefType abstract_memref(Type t);

    static Value concretize_tensor(OpBuilder b, Location l, Type t, Value v);

    static Value abstractize_tensor(OpBuilder b, Location l, Type t, Value v);

    static std::string printable(Type t);
  };

  // This ValueHash is used in several places: ClockAnalysis.h.
  // CondToPred.h. EnsureDominance.h. Hence, it's better to
  // extract it here.
  struct ValueHash {
    template <class Value>
    std::size_t operator() (const Value& value) const {
      // IR/Value.h
      //::llvm::hash_code hash_value(Value value) can be used
      return hash_value(value);
      // return std::hash<void*>()(value.getAsOpaquePointer());
    }
  };

  struct TypeHash {
    template < class Type >
    std::size_t operator() (const Type& type) const {
      return (std::size_t) type.getAsOpaquePointer();
    }
  };
}

#endif
