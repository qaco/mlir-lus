// OnClockOp class definitions -*- C++ -*- //

#ifndef MLIRLUS_ON_CLOCK_OP_H
#define MLIRLUS_ON_CLOCK_OP_H

#include "Clocking/ClassicClock.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "ClockType.h"
#include "Yield.h"

namespace mlir {
  namespace lus {

    class OnClockOp : public Op <
      OnClockOp,
      OpTrait::OneRegion,
      OpTrait::VariadicOperands,
      OpTrait::VariadicResults,
      OpTrait::ZeroSuccessor,
      RegionBranchOpInterface::Trait,
      OpTrait::SingleBlockImplicitTerminator<lus::YieldOp>::Impl,
      OpTrait::HasRecursiveSideEffects,
      OpTrait::NoRegionArguments> {
    public:
      
      using Op<
      OnClockOp,
      OpTrait::OneRegion,
      OpTrait::VariadicOperands,
      OpTrait::VariadicResults,
      OpTrait::ZeroSuccessor,
      RegionBranchOpInterface::Trait,
      OpTrait::SingleBlockImplicitTerminator<lus::YieldOp>::Impl,
      OpTrait::HasRecursiveSideEffects,
      OpTrait::NoRegionArguments>::Op;

      using operand_range = OperandRange;
      using operand_iterator = operand_range::iterator;
      
      void getSuccessorRegions(::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
      
      static StringRef getFlagsAttrName() { return "flags"; }
      
      static ArrayRef<llvm::StringRef> getAttributeNames() {
	static llvm::StringRef attrNames[] = {getFlagsAttrName()};
	return llvm::makeArrayRef(attrNames);
      }

      Region& getBody() { return getOperation()->getRegion(0); }
      
      static StringRef getOperationName() { return "lus.on_clock"; }

      Operation *nested();
      ClassicClock eqClock();
      void clockedOutputs(SmallVectorImpl<ClassicClock> &outputs);

      static void build(Builder &b, OperationState &s,
			ClassicClock eqClock,
			SmallVectorImpl<ClassicClock>& outputClocks,
			result_type_range resultTypes);
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
