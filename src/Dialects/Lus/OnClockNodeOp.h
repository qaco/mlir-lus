// OnClockNodeOp class definitions -*- C++ -*- //

#ifndef MLIRLUS_ON_CLOCK_NODE_OP_H
#define MLIRLUS_ON_CLOCK_NODE_OP_H

#include "Clocking/ClassicClock.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "ClockType.h"
#include "Yield.h"

namespace mlir {
  namespace lus {

    class OnClockNodeOp : public Op <
      OnClockNodeOp,
      OpTrait::VariadicOperands,
      OpTrait::ZeroResult,
      OpTrait::ZeroSuccessor,
      OpTrait::HasRecursiveSideEffects> {
    public:
      
      using Op<
      OnClockNodeOp,
      OpTrait::VariadicOperands,
      OpTrait::ZeroResult,
      OpTrait::ZeroSuccessor,
      OpTrait::HasRecursiveSideEffects>::Op;

      using operand_range = OperandRange;
      using operand_iterator = operand_range::iterator;
          
      static StringRef getFlagsAttrName() { return "flags"; }
      
      static ArrayRef<llvm::StringRef> getAttributeNames() {
	static llvm::StringRef attrNames[] = {getFlagsAttrName()};
	return llvm::makeArrayRef(attrNames);
      }

      static StringRef getOperationName() { return "lus.on_clock_node"; }

      // void clockedInputs(SmallVectorImpl<ClassicClock> &inputs);
      void clockedOutputs(SmallVectorImpl<ClassicClock> &outputs);

      static void build(Builder &b, OperationState &s,
			SmallVectorImpl<ClassicClock>& inputClocks,
			SmallVectorImpl<ClassicClock>& outputClocks);
      LogicalResult verify();
      static ParseResult parse(OpAsmParser &parser,
			       OperationState &result);
      void print(OpAsmPrinter &p);

      Type getFlagsType() {
	return getFlagsAttr().getValue();
      }
      
    private:
      
      TypeAttr getFlagsAttr() {
	return getOperation()->getAttrOfType<TypeAttr>(getFlagsAttrName());
      }

      std::vector<bool> getFlags() {
	return getFlagsAttr().getValue().cast<ClockType>().getSeq();
      }
    };
  }
}

#endif
