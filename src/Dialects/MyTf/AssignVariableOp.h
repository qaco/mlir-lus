// -*- C++ -*- //

#ifndef ASSIGNVARIABLE_OP_H
#define ASSIGNVARIABLE_OP_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

namespace mlir {
  namespace mytf {

    class AssignVariableOp: public Op <
      AssignVariableOp,
      OpTrait::ZeroResult,
      OpTrait::NOperands<2>::Impl,
      OpTrait::ZeroSuccessor > {

    public:

      using Op::Op;

      static StringRef getOperationName() {
	return "tf.AssignVariableOp";
      }

      static void build(Builder &builder, OperationState &state,
			Value resource, Value value);
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
