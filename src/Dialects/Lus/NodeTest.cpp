#include "NodeTest.h"
#include "ClockAnnotation.h"
#include "OnClockNodeOp.h"
#include "../Sync/InputOp.h"
#include "mlir/IR/BlockAndValueMapping.h"

namespace mlir {
  namespace lus {

    // Misc IOs
    
    TypeAttr NodeTestOp::getTypeAttr() {
      return getOperation()->getAttrOfType<TypeAttr>(typeAttrName());
    }
    
    FunctionType NodeTestOp::getType() {
      return getTypeAttr().getValue().cast<FunctionType>();
    }
    
    MutableArrayRef<BlockArgument> NodeTestOp::getArgs() {
      return getBody().front().getArguments();
    }

    // Inputs

    unsigned NodeTestOp::getNumInputs() {
      return getType().getInputs().size();
    }
    
    ArrayRef<Type> NodeTestOp::getInputTypes() {
      return getType().getInputs();
    }
    
    BlockArgument NodeTestOp::getInputValue(unsigned i) {
	assert(i < getNumInputs());
	return getBody().getArgument(i);
    }
    
    iterator_range<Block::args_iterator> NodeTestOp::getInputs() {
      if (outputsExplicit())
	return getBody().getArguments()
	  .drop_back(getCardState()+getNumOutputs());
      else
	return getBody().getArguments()
	  .drop_back(getCardState());
    }

    // State

    int NodeTestOp::getCardState() {
      return
	getOperation()
	->getAttr(cardStateAttrName()).cast<IntegerAttr>().getInt();
    }
    
    ArrayRef<Type> NodeTestOp::getStateTypes() {
      SmallVector<Type> t;
      for (int i = 0; i < getCardState(); i++)
	t.push_back(getStateValue(i).getType());
      return t;
    }
    
    BlockArgument NodeTestOp::getStateValue(unsigned i) {
      int to_drop = getBody().getNumArguments() - getCardState();
      return getBody().getArgument(to_drop + i);
    }
    
    iterator_range<Block::args_iterator> NodeTestOp::getState() {
      int to_drop = getBody().getNumArguments() - getCardState();
      return getArgs().drop_front(to_drop);
    }
    
    Value NodeTestOp::addState(OpBuilder &b, Type type) {
      Value v = getBody().addArgument(type);
      getAuxRegion().addArgument(type);
      IntegerAttr stateAttr = IntegerAttr::get(b.getI32Type(),
					       getCardState() + 1);
      getOperation()->setAttr(cardStateAttrName(),stateAttr);
      return v;
    }

    // Outputs
    
    unsigned NodeTestOp::getNumOutputs() {
      return getType().getResults().size();
    }
    
    ArrayRef<Type> NodeTestOp::getOutputTypes() {
      return getType().getResults();
    }
    
    BlockArgument NodeTestOp::getResultValue(unsigned i) {
      assert(outputsExplicit());
      assert(i < getNumOutputs());
      return getBody().getArgument(getNumInputs() + i);
    }
    
    iterator_range<Block::args_iterator> NodeTestOp::getOutputs() {
      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stderr", err);

      if (!outputsExplicit()) {
	stream << "\n(" << sym_name() << ") "
	       << "num region arguments: "
	       << getBody().getNumArguments() << "\n"
	       << "num inputs: "
	       << getNumInputs() << "\n"
	       << "num outputs: "
	       << getNumOutputs() << "\n"
	       << "num state: "
	       << getCardState() << "\n";
	assert(false);
      }
      assert(outputsExplicit());
      return getBody().getArguments()
	.drop_front(getNumInputs())
	.drop_back(getCardState());
    }

    bool NodeTestOp::outputsExplicit() {
      return (getBody().getNumArguments()
	      == getNumInputs() + getCardState() + getNumOutputs());
    }

    void NodeTestOp::makeOutputsExplicit() {
      assert(!outputsExplicit());
      unsigned offset = 0;
      for (Type ty: getOutputTypes()) {
	getBody().insertArgument(getNumInputs() + offset, ty);
	getAuxRegion().insertArgument(getNumInputs() + offset, ty);
	offset++;
      }
    }

    // Region management

    void NodeTestOp::forceDominance() {
      assert(!isDominanceOn());
      OpBuilder b(getOperation());
      Region& graphRegion = getBody();
      int64_t dom = 1;
      IntegerAttr domAttr = IntegerAttr::get(b.getIntegerType(1),dom);
      getOperation()->setAttr(domAttrName(), domAttr);
      Region& SSACFGRegion = getBody();
      SSACFGRegion.takeBody(graphRegion);
    }
    
    bool NodeTestOp::isDominanceOn() {
      return
	getOperation()
	->getAttr(domAttrName()).cast<IntegerAttr>().getInt();
    }
      
    Region& NodeTestOp::getBody() {
      unsigned index = 2;
      if (isDominanceOn())
	index = 1;
      return getOperation()->getRegion(index);
    }

    Region& NodeTestOp::getAuxRegion() {
      return getOperation()->getRegion(0);
    }
      
    Region* NodeTestOp::getCallableRegion() {
      return &getBody();
    }

    ArrayRef<Type> NodeTestOp::getCallableResults() {
      return getType().getResults();
    }

    bool NodeTestOp::isClocked() {
      unsigned num = getAuxRegion().front().getOperations().size();
      // std::error_code err;
      // llvm::raw_fd_ostream stream("/dev/stdout", err);
      // stream << "\n" << sym_name() << " : " << num << "\n";
      return num == 2;
    }

    OnClockNodeOp NodeTestOp::getSignatureClock() {
      Operation *op = &getAuxRegion().front().front();
      return dyn_cast<OnClockNodeOp>(op);
    }

    void NodeTestOp::clockedInputs(SmallVectorImpl<ClassicClock> &inputs) {
      OnClockNodeOp onClockOp = getSignatureClock();
      FunctionType ft = onClockOp.getFlagsType().dyn_cast<FunctionType>();
      operand_iterator it = onClockOp.getOperation()->operand_begin();

      for (auto ty : ft.getInputs()) {
	SmallVector<Subsampling,4> subsamplings;
	ClockType cty = ty.cast<ClockType>();
	std::vector<bool> flags = cty.getSeq();
	for (bool flag: flags) {
	  BlockArgument ba = (*(it++)).cast<BlockArgument>();
	  Value operand = getBody().front().getArgument(ba.getArgNumber());

	  if (operand.hasOneUse()
	      && isa<sync::InputOp>(*operand.user_begin())) {
	    sync::InputOp iOp= dyn_cast<sync::InputOp>(*operand.user_begin());
	    operand = iOp.getResult();
	  }
	  
	  Subsampling ss(flag,operand);
	  subsamplings.push_back(ss);
	}
	ClassicClock cc(ClassicClock::PREFIX_BASE,subsamplings);
	inputs.push_back(cc);
      }
    }

    void NodeTestOp::clockedOutputs(SmallVectorImpl<ClassicClock> &outputs) {
      OnClockNodeOp onClockOp = getSignatureClock();
      FunctionType ft = onClockOp.getFlagsType().dyn_cast<FunctionType>();

      unsigned offset = 0;
      for (auto ty : ft.getInputs()) {
	ClockType cty = ty.cast<ClockType>();
	std::vector<bool> flags = cty.getSeq();
	offset += flags.size();
      }

      operand_iterator it = onClockOp.getOperation()->operand_begin() + offset;

      for (auto ty : ft.getResults()) {
	SmallVector<Subsampling,4> subsamplings;
	ClockType cty = ty.cast<ClockType>();
	std::vector<bool> flags = cty.getSeq();
	for (bool flag: flags) {
	  BlockArgument ba = (*(it++)).cast<BlockArgument>();
	  Value operand = getBody().front().getArgument(ba.getArgNumber());
	  // ba is an output
	  if (ba.getArgNumber() >= getNumInputs()) {
	    YieldOp yieldOp = getYield();
	    operand = yieldOp.getOutputs()[(ba.getArgNumber() - getNumInputs())];
	  }
	  // ba is a sampled input
	  if (operand.hasOneUse()
	      && isa<sync::InputOp>(*operand.user_begin())) {
	    sync::InputOp iOp= dyn_cast<sync::InputOp>(*operand.user_begin());
	    operand = iOp.getResult();
	  }
    	  Subsampling ss(flag,operand);
    	  subsamplings.push_back(ss);
	}
	ClassicClock cc(ClassicClock::PREFIX_BASE,subsamplings);
	outputs.push_back(cc);
      }
    }

    RegionKind NodeTestOp::getRegionKind(unsigned index) {
      switch(index){
      case 0: return RegionKind::SSACFG ;
      case 1: return RegionKind::SSACFG ;
      case 2: return RegionKind::Graph ;
      default: assert(false) ;
      }
    }

    YieldOp NodeTestOp::getYield() {
      return dyn_cast<YieldOp>(getBody().back().getTerminator());
    }

    // Main algorithmics
    
    void NodeTestOp::print(OpAsmPrinter &p) {
      ArrayRef<Type> argTypes = getType().getInputs();
      ArrayRef<Type> resultTypes = getType().getResults();
      auto funcName = sym_name();
      
      p << ' ';

      if (isDominanceOn()) {
	p << "dom ";
      }
      
      p.printSymbolName(funcName);

      p << '(';
      for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
	if (i > 0)
	  p << ", ";
	p.printRegionArgument(getInputValue(i), {});
      }
      p << ") ";
      if (getCardState() > 0) {
	p << "state (";
	for (unsigned i = 0, e = getCardState(); i < e; ++i) {
	  if (i > 0)
	    p << ", ";
	  p.printRegionArgument(getStateValue(i), {});
	}
	p << ") ";
      }
      
      p << "-> (";
      for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
	if (i > 0)
	  p << ", ";
	if (outputsExplicit()) {
	  p.printRegionArgument(getResultValue(i));
	}
	else {
	  p.printType(getOutputTypes()[i]);
	}
      }
      p << ')';

      if (isClocked()) {
	p << " clock { ";
	SmallVector<StringRef> defaultDialectStack{"builtin"};
	Operation *op = &getAuxRegion().front().front();
	auto *opInfo = op->getAbstractOperation();
	opInfo->printAssembly(op, p, defaultDialectStack.back());
	p << "}";
      }

      p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
		    /*printBlockTerminators=*/true);
      
    }

    ParseResult NodeTestOp::parse(OpAsmParser &parser,
				  OperationState &result) {
      SmallVector<OpAsmParser::OperandType, 4> inputArgs;
      SmallVector<OpAsmParser::OperandType, 4> resultArgs;
      SmallVector<OpAsmParser::OperandType, 4> stateArgs;
      SmallVector<NamedAttrList, 4> argAttrs;
      SmallVector<NamedAttrList, 4> resultAttrs;
      SmallVector<NamedAttrList, 4> stateAttrs;
      bool isVariadic = false;
      bool allowAttributes = false;
      bool allowVariadic = false;
      SmallVector<Type, 4> inputTypes;
      SmallVector<Type, 4> resultTypes;
      SmallVector<Type, 4> stateTypes;
      auto &builder = parser.getBuilder();

      int64_t dom = 0;
      if (succeeded(parser.parseOptionalKeyword("dom")))
	dom = 1;
      IntegerAttr domAttr = IntegerAttr::get(builder.getIntegerType(1),
					     dom);
      result.addAttribute(domAttrName(), domAttr);
      
      // Parse the name as a symbol.
      StringAttr nameAttr;
      if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
				 result.attributes))
	return failure();

      // Parse the function interface

      function_like_impl::parseFunctionArgumentList
	(parser, allowAttributes, allowVariadic,
	 inputArgs, inputTypes, argAttrs, isVariadic);

      if (succeeded(parser.parseOptionalKeyword("state"))) {
	function_like_impl::parseFunctionArgumentList
	(parser, allowAttributes, allowVariadic,
	 stateArgs, stateTypes, stateAttrs, isVariadic);
      }

      IntegerAttr stateAttr = IntegerAttr::get(builder.getI32Type(),
					       stateArgs.size());
      result.addAttribute(cardStateAttrName(), stateAttr);
      
      if (parser.parseArrow())
	return failure();

      function_like_impl::parseFunctionArgumentList
	(parser, allowAttributes, allowVariadic,
	 resultArgs, resultTypes, resultAttrs, isVariadic);

      // Build the function type
      
      FunctionType type = builder.getFunctionType(inputTypes,
						  resultTypes);
      result.addAttribute(typeAttrName(), TypeAttr::get(type));
      
      // Parse the node bodies.

      SmallVector<OpAsmParser::OperandType, 4> allParams;
      allParams.append(inputArgs);
      allParams.append(resultArgs);
      allParams.append(stateArgs);
      SmallVector<Type, 4> allTypes;
      allTypes.append(inputTypes);
      allTypes.append(resultTypes);
      allTypes.append(stateTypes);

      auto *auxReg = result.addRegion();
      auto *bodyDom = result.addRegion();
      auto *bodyFree = result.addRegion();
      Region *body = bodyFree;
      if (dom) {
	body = bodyDom;
      }
      
      // Parse the clock type
      
      if (succeeded(parser.parseOptionalKeyword("clock"))) {
	
	llvm::SMLoc loc = parser.getCurrentLocation();
	if (parser.parseRegion(*auxReg,
			       allParams,
			       allParams.empty() ? ArrayRef<Type>():allTypes,
			       /*enableNameShadowing=*/false))
	  return failure();
	if (auxReg->front().getOperations().size() == 1
	    && !isa<OnClockNodeOp>(auxReg->front().front()))
	  return parser.emitError(loc,
				  "clock region expected OnClockNodeOp");
      }
      else {
	unsigned diff = allTypes.size() - allParams.size();
	for (unsigned i = 0; i < diff; i++)
	  allTypes.pop_back();
      }
      NodeTestOp::ensureTerminator(*auxReg, builder, result.location);

      // Parse the node body
      
      llvm::SMLoc loc = parser.getCurrentLocation();
      if (parser.parseRegion(*body,
			     allParams,
			     allParams.empty() ? ArrayRef<Type>() : allTypes,
			     /*enableNameShadowing=*/false))
	return failure();
      
      if (body->empty())
	  return parser.emitError(loc, "expected non-empty node body");

      if (auxReg->getNumArguments() < body->getNumArguments()) {
	auxReg->addArguments(body->getArgumentTypes());
      }

      return success();
    }

    LogicalResult NodeTestOp::verify() {

      if (!outputsExplicit())
	makeOutputsExplicit();
      
      {
    	auto tblgen_sym_name = (*this)->getAttr(sym_nameAttrName());
    	if (!tblgen_sym_name)
    	  return emitOpError("requires attribute 'sym_name'");
  
    	if (tblgen_sym_name && !((tblgen_sym_name.isa<StringAttr>())))
    	  return emitOpError("attribute 'sym_name' failed to satisfy constraint: string attribute");
      }
      {
    	auto tblgen_type = (*this)->getAttr(typeAttrName());
    	if (!tblgen_type)
    	  return emitOpError("requires attribute 'type'");
  
    	if (tblgen_type && !(((tblgen_type.isa<TypeAttr>())) && ((tblgen_type.cast<TypeAttr>().getValue().isa<Type>()))))
    	  return emitOpError("attribute 'type' failed to satisfy constraint: any type attribute");
      }
      {
	unsigned index = 0; (void)index;
	for (Region &region : MutableArrayRef<Region>((*this)->getRegion(0))) {
	  (void)region;
	  if (!((true))) {
	    return emitOpError("region #") << index << " ('body') failed to verify constraint: any region";
	  }
	  ++index;
	}
      }
      
      // Verify that the argument list of the function and the arg list of the entry
      // block line up.  The trait already verified that the number of arguments is
      // the same between the signature and the block.
      // auto fnInputTypes = getType().getInputs();
      // Block &entryBlock = getBody().front();
      // for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
      // 	if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      // 	  return emitOpError("type of entry block argument #")
      // 	    << i << '(' << entryBlock.getArgument(i).getType()
      // 	    << ") must match the type of the corresponding argument in "
      // 	    << "function signature(" << fnInputTypes[i] << ')';

      return success();
    }
  }
}
