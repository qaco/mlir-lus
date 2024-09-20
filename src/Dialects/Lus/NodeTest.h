// -*- C++ -*- //

#ifndef NODE_TEST_OP_H
#define NODE_TEST_OP_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/RegionKindInterface.h" 
#include "Yield.h"
#include "OnClockNodeOp.h"

namespace mlir {
  namespace lus {

    class NodeTestOp: public Op <
      NodeTestOp,
      OpTrait::ZeroResult,
      OpTrait::ZeroSuccessor,
      OpTrait::ZeroOperands,
      CallableOpInterface::Trait,
      OpTrait::IsIsolatedFromAbove,
      OpTrait::NRegions<3>::Impl,
      OpTrait::SingleBlockImplicitTerminator<lus::YieldOp>::Impl,
      RegionKindInterface::Trait,
      SymbolOpInterface::Trait> {

    public:
      
      using Op <NodeTestOp,
		OpTrait::ZeroResult,
		OpTrait::ZeroSuccessor,
		OpTrait::ZeroOperands,
		CallableOpInterface::Trait,
		OpTrait::IsIsolatedFromAbove,
		OpTrait::NRegions<3>::Impl,
		OpTrait::SingleBlockImplicitTerminator<lus::YieldOp>::Impl,
		RegionKindInterface::Trait,
		SymbolOpInterface::Trait>::Op;

      static constexpr llvm::StringLiteral getOperationName() {
	return llvm::StringLiteral("lus.node_test");
      }
      
      // Attributes management

      static StringRef cardStateAttrName() { return "card_state"; }
      
      static StringRef domAttrName() { return "dom"; }

      static StringRef typeAttrName() { return "type"; }

      static StringRef sym_nameAttrName() { return "sym_name"; }
      
      static ArrayRef<llvm::StringRef> getAttributeNames() {
	static llvm::StringRef attrNames[] = {sym_nameAttrName(),
					      typeAttrName(),
					      domAttrName(),
					      cardStateAttrName()};
	return llvm::makeArrayRef(attrNames);
      }
      
      StringRef sym_name() {
	return
	  getOperation()
	  ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
	  .getValue();
      }

      // IOs

      // Misc
      TypeAttr getTypeAttr();
      FunctionType getType();
      MutableArrayRef<BlockArgument> getArgs();

      // Inputs
      unsigned getNumInputs();
      ArrayRef<Type> getStaticTypes() { return {}; }
      ArrayRef<Type> getInputTypes();
      BlockArgument getInputValue(unsigned i);
      iterator_range<Block::args_iterator> getInputs();

      // State
      int getCardState();
      ArrayRef<Type> getStateTypes();
      BlockArgument getStateValue(unsigned i);
      iterator_range<Block::args_iterator> getState();
      Value addState(OpBuilder &b, Type type);

      // Outputs
      unsigned getNumOutputs();
      ArrayRef<Type> getOutputTypes();
      BlockArgument getResultValue(unsigned i);
      iterator_range<Block::args_iterator> getOutputs();
      bool outputsExplicit();
      void makeOutputsExplicit();

      // Regions management

      void forceDominance();
      bool isDominanceOn();
      Region& getBody();
      bool hasBody() { return !getBody().empty() ; }
      Region& getAuxRegion();
      Region *getCallableRegion();
      ArrayRef<Type> getCallableResults();
      static RegionKind getRegionKind(unsigned index);
      YieldOp getYield();

      // Clocking

      bool isClocked();
      void clockedInputs(SmallVectorImpl<ClassicClock> &inputs);
      void clockedOutputs(SmallVectorImpl<ClassicClock> &outputs);
      OnClockNodeOp getSignatureClock();

      // Main algorithmics

      static ParseResult parse(OpAsmParser &parser, OperationState &result);
      void print(OpAsmPrinter &p);
      LogicalResult verify();

    };
  }
}

#endif
