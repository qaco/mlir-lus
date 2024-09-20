// -*- C++ -*- //

#ifndef MLIRLUS_INSTANCE_TEST_H
#define MLIRLUS_INSTANCE_TEST_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "NodeTest.h"

namespace mlir {
  namespace lus {

    class InstanceTestOp : public Op <
      InstanceTestOp,
      OpTrait::VariadicResults,
      OpTrait::ZeroSuccessor,
      CallOpInterface::Trait,
      OpTrait::VariadicOperands > {
    public:
      using Op::Op;

      static StringRef getOperationName() { return "lus.instance_test"; }
      /// Required by CallOpInterface
      CallInterfaceCallable getCallableForCallee() { return getCallee(); }
      /// The name of the node we intend to instantiate
      StringRef getCalleeName() { return getCallee().getValue() ; }
      /// The node we intend to instantiate
      NodeTestOp getCalleeNode();

      operand_range getArgOperands() {return {operand_begin(),operand_end()};}
      
      LogicalResult verify();
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      bool mustInline();
      static ArrayRef<StringRef> getAttributeNames();
    private:
      FlatSymbolRefAttr getCallee();
      static StringRef getInlineAttrName() { return "inline"; }
    };
  }
}

#endif
