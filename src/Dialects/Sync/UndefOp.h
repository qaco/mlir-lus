// -*- C++ -*- //

#ifndef SYNC_UNDEF_OP_H
#define SYNC_UNDEF_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

namespace mlir {
  namespace sync {

    class UndefOp: public Op <
      UndefOp,
      OpTrait::OneResult,
      OpTrait::ZeroOperands,
      OpTrait::ZeroSuccessor > {

    public:
      
      using Op::Op;

      static StringRef getOperationName() { return "sync.undef"; }

      static void build(Builder &odsBuilder,
			OperationState &odsState,
			Type t);
      
      static ParseResult parse(OpAsmParser &parser, OperationState &result);

      void print(OpAsmPrinter &p);

      LogicalResult verify();

      static ArrayRef<StringRef> getAttributeNames() { return {}; }
    };
  }
}

#endif
