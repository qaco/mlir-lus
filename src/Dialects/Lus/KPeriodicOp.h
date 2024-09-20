// KperiodicOp class definitions -*- C++ -*- //

#ifndef MLIRLUS_KPERIODIC_H
#define MLIRLUS_KPERIODIC_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

namespace mlir {
  namespace lus {

    class KperiodicOp : public Op <
      KperiodicOp,
      OpTrait::OneResult,
      OpTrait::ZeroSuccessor,
      OpTrait::ZeroOperands > {
    public:
      using Op::Op;

      static StringRef getOperationName() { return "lus.kperiodic"; }

      static void build(Builder &odsBuilder,
			OperationState &odsState,
			StringRef word);
      
      StringRef getWord() {
	Operation *op = getOperation();
	return op->getAttrOfType<StringAttr>(getWordAttrName()).strref();
      }

      std::vector<bool> getPrefix();
      std::vector<bool> getPeriod();
      bool isPair() {
	return getPrefix().size() == 1 && getPeriod().size() == 1;
      }
      bool isHeadLess() {
	return getPrefix().size() == 0;
      }
      bool isSingleton() {
	return isHeadLess() && getPeriod().size() == 1;
      }

      LogicalResult verify();
      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      static ArrayRef<StringRef> getAttributeNames() { return {}; }
    private:
      static StringRef getWordAttrName() { return "word"; }
    };
  }
}

#endif
