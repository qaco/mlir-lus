// -*- C++ -*- //

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#ifndef LUS_OUTPUT_OP_H
#define LUS_OUTPUT_OP_H

namespace mlir {
  namespace lus {

    class OutputOp: public Op < OutputOp,
				OpTrait::OneResult,
				OpTrait::OneOperand,
				OpTrait::ZeroSuccessor > {

      public:

      using Op::Op;

      static StringRef getOperationName() { return "lus.output"; }

      int64_t getPosition();

      static void build(OpBuilder &builder, OperationState &result,
			int64_t pos, Value v);

      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      
      LogicalResult verify() ;
      
      void print(OpAsmPrinter &p);

      static ArrayRef<StringRef> getAttributeNames();

    private:

      static StringRef getPosKey() { return "pos"; }
    };
    
  }
}

#endif
