// -*- C++ -*- //

#ifndef VARHANDLEOP_H
#define VARHANDLEOP_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace mytf {

    class VarHandleOp : public Op <VarHandleOp,
				   OpTrait::OneResult,
				   OpTrait::ZeroOperands,
				   OpTrait::ZeroSuccessor > {

    public:

      using Op::Op;

      static StringRef getOperationName() { return "tf.VarHandleOp"; }

      StringRef getSharedName();

      static void build(OpBuilder &builder, OperationState &result,
			StringRef sharedName, Type t);

      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      
      static ArrayRef<StringRef> getAttributeNames();
      
      LogicalResult verify() ;
      
      void print(OpAsmPrinter &p);

    private:

      static StringRef getSharedNameKey() { return "shared_name"; }
    }; 
  }
}


#endif
