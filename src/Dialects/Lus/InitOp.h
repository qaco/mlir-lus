// -*- C++ -*- //

#ifndef INIT_OP_H
#define INIT_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

namespace mlir {
  namespace lus {

    class InitOp: public Op <
      InitOp,
      OpTrait::NOperands<2>::Impl,
      OpTrait::OneResult,
      OpTrait::ZeroSuccessor > {

    public:

      using Op::Op;

      static StringRef getOperationName() { return "lus.macro_merge_kp01"; }
      static void build(Builder &odsBuilder,OperationState &odsState,
			Value v, Value i);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();
      static ArrayRef<StringRef> getAttributeNames() { return {}; }

    };
  }
}
    
#endif
