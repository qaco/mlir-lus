#include "Helpers.h"

namespace mlir {

  void Helpers::feed_with_deps_of(Value v,
				  llvm::SmallSet<Operation*,4> &deps) {
    Operation *defOp = v.getDefiningOp();
    deps.insert(defOp);
    for (Value arg: defOp->getOperands()) {
      feed_with_deps_of(arg, deps);
    }
  }

  bool Helpers::depends_on_inputs(Value v) {
    if (auto op = v.getDefiningOp()) {
      bool res = false ;
      for (Value arg: op->getOperands()) {
	res = res || depends_on_inputs(arg);
      }
      return res;
    }
    else {
      return true;
    }
  }

  TensorType Helpers::abstract_tensor(Type t) {
    if (t.isa<TensorType>()) {
      TensorType tt = t.cast<TensorType>();
      std::vector<int64_t> shape(tt.getShape().size());
      for (unsigned i = 0; i < tt.getShape().size(); i++)
	shape[i] = -1;
      return RankedTensorType::get(shape, tt.getElementType());
    }
    else {
      std::vector<int64_t> shape(1);
      shape[0] = -1;
      return RankedTensorType::get(shape, t);
    }
  }

  MemRefType Helpers::abstract_memref(Type t) {
    if (t.isa<TensorType>()) {
      TensorType tt = t.cast<TensorType>();
      std::vector<int64_t> shape(tt.getShape().size());
      for (unsigned i = 0; i < tt.getShape().size(); i++)
	shape[i] = -1;
      return MemRefType::get(shape, tt.getElementType());
    }
    else {
      std::vector<int64_t> shape(1);
      shape[0] = -1;
      return MemRefType::get(shape, t);
    }
  }

  Value Helpers::concretize_tensor(OpBuilder b, Location l, Type t, Value v) {
    if (t.isa<TensorType>()) {
      tensor::CastOp op = b.create<tensor::CastOp>(l,t,v);
      return op.getResult();
    }
    else {
      Attribute az = IntegerAttr::get(b.getIndexType(), 0);
      arith::ConstantOp oz = b.create<arith::ConstantOp>(l, az);
      std::vector<int64_t> shape(1);
      shape[0] = 1;
      TensorType ot = RankedTensorType::get(shape, t);
      tensor::CastOp op = b.create<tensor::CastOp>(l,ot,v);
      tensor::ExtractOp op2 = b.create<tensor::ExtractOp>(l,
							  op.getResult(),
							  oz.getResult());
      return op2.getResult();
    }
  }

  Value Helpers::abstractize_tensor(OpBuilder b, Location l,
				    Type t, Value v) {
    if (v.getType().isa<TensorType>()) {
      tensor::CastOp op = b.create<tensor::CastOp>(l,t,v);
      return op.getResult();
    }
    else {
      tensor::FromElementsOp op = b.create<tensor::FromElementsOp>
	(l,v);
      std::vector<int64_t> shape(1);
      shape[0] = -1;
      TensorType nt = RankedTensorType::get(shape, v.getType());
      tensor::CastOp op2 = b.create<tensor::CastOp>(l,nt,op.getResult());
      return op2.getResult();
    }
  }

  std::string Helpers::printable(Type t) {
    std::string name;
    llvm::raw_string_ostream stream(name);
    stream << t;
    name.erase(std::remove(name.begin(), name.end(), '<'), name.end());
    name.erase(std::remove(name.begin(), name.end(), '>'), name.end());
    name.erase(std::remove(name.begin(), name.end(), '?'), name.end());
    return name;
  }

}
