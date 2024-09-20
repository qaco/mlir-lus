// -*- C++ -*- //

#ifndef INPUT_OP_H
#define INPUT_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

namespace mlir {
  namespace sync {

    class InputOp: public Op <
      InputOp,
      OpTrait::OneResult,
      OpTrait::OneOperand,
      OpTrait::ZeroSuccessor > {

    public:

      using Op::Op;

      static StringRef getOperationName() { return "sync.input"; }
      static void build(Builder &, OperationState &, Value,
			bool isSignal = true);
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();

      /// Get the input signal
      Value getSignal() { return getOperand(); }

      static ArrayRef<StringRef> getAttributeNames() { return {}; }

    private:
      static StringRef getSigAttrName() { return "isSignal"; }
      static int64_t isSigValue() {return 1;}
      bool isSig() {
	Operation *op = getOperation();
	int64_t is=op->getAttrOfType<IntegerAttr>(getSigAttrName()).getInt();
	return is == isSigValue();
      }
    };
  }
}

#endif
