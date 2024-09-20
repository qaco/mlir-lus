// FbyOp class definitions -*- C++ -*- //

#ifndef MLIRLUS_FBY_H
#define MLIRLUS_FBY_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace lus {

    class FbyOp : public Op <
      FbyOp,
      OpTrait::OneResult,
      OpTrait::ZeroSuccessor,
      OpTrait::SameOperandsAndResultShape,
      OpTrait::NOperands<2>::Impl> {
    public:
      using Op::Op;

      static StringRef getOperationName() { return "lus.fby"; }

      static void build(Builder &odsBuilder,
			OperationState &odsState,
			Value l, Value r);
      
      Value getLhs() { return getOperand(0) ; }
      Value getRhs() { return getOperand(1) ; }

      LogicalResult verify();
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      static ArrayRef<StringRef> getAttributeNames() { return {}; }
    };
  }
}

#endif
