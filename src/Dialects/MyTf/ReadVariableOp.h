// -*- C++ -*- //

#ifndef READVARIABLE_OP_H
#define READVARIABLE_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

namespace mlir {
  namespace mytf {

    class ReadVariableOp: public Op <
      ReadVariableOp,
      OpTrait::OneResult,
      OpTrait::OneOperand,
      OpTrait::ZeroSuccessor > {

    public:

      using Op::Op;

      static StringRef getOperationName() {
	return "tf.ReadVariableOp";
      }

      static void build(Builder &builder, OperationState &state,
			Value resource);
      static ParseResult parse(OpAsmParser &parser,
			       OperationState &result);

      static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
	return {};
      }
      void print(OpAsmPrinter &p);
      LogicalResult verify();
    };
  }
}

#endif
