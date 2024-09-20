#include "OnClockOp.h"
#include "Yield.h"
#include "llvm/ADT/StringExtras.h"
#include "ClockAnnotation.h"

namespace mlir {
  namespace lus {

    using operand_range = OperandRange;
    using operand_iterator = operand_range::iterator;
    
    void OnClockOp::getSuccessorRegions(Optional<unsigned> index,
					ArrayRef<Attribute> operands,
					SmallVectorImpl<RegionSuccessor> &regions) {
  regions.push_back(RegionSuccessor(getResults()));
      // regions.push_back(getBody().getParentRegion());
}

    Operation *OnClockOp::nested() {
      return &getBody().front().front();
    }

    ClassicClock OnClockOp::eqClock() {
      FunctionType ft = getFlagsType().dyn_cast<FunctionType>();
      SmallVector<Subsampling,4> subsamplings;
      ClockType cty = ft.getInput(0).cast<ClockType>();
      std::vector<bool> flags = cty.getSeq();
      for (unsigned i = 0; i < flags.size(); i++) {
	bool flag = flags[i];
	Value operand = getOperand(i);
	Subsampling ss(flag,operand);
	subsamplings.push_back(ss);
      }
      ClassicClock cc(ClassicClock::PREFIX_BASE,subsamplings);
      return cc;
    }

    void OnClockOp::clockedOutputs(SmallVectorImpl<ClassicClock> &outputs) {
      FunctionType ft = getFlagsType().dyn_cast<FunctionType>();

      unsigned offset = 0;
      for (auto ty : ft.getInputs()) {
	ClockType cty = ty.cast<ClockType>();
	std::vector<bool> flags = cty.getSeq();
	offset += flags.size();
      }

      operand_iterator it = getOperation()->operand_begin() + offset;

      for (auto ty : ft.getResults()) {
	SmallVector<Subsampling,4> subsamplings;
	ClockType cty = ty.cast<ClockType>();
	std::vector<bool> flags = cty.getSeq();
	for (bool flag: flags) {
	  Value operand = *(it++);
	  Subsampling ss(flag,operand);
	  subsamplings.push_back(ss);
	}
	ClassicClock cc(ClassicClock::PREFIX_BASE,subsamplings);
	outputs.push_back(cc);
      }
    }

    void OnClockOp::build(Builder &b, OperationState &s,
			  ClassicClock eqClock,
			  SmallVectorImpl<ClassicClock>& outputClocks,
			  result_type_range resultTypes) {
      // Input clocks
      std::vector<bool> eqFlags;
      for (auto ss: eqClock.getSubsamplings()) {
	s.addOperands({ss.data});
	eqFlags.push_back(ss.on_or_onnot);
      }
      Type eqTy = b.getType<ClockType,std::vector<bool>>(eqFlags);

      // Output clocks
      SmallVector<Type,4> outputFlags;
      for (ClassicClock cc: outputClocks) {
	std::vector<bool> flags;
	for (auto ss: cc.getSubsamplings()) {
	  s.addOperands({ss.data});
	  flags.push_back(ss.on_or_onnot);
	}
	Type t = b.getType<ClockType,std::vector<bool>>(flags);
	outputFlags.push_back(t);
      }
      // Full clock
      Type ft = b.getFunctionType({eqTy},outputFlags);
      s.addAttribute(getFlagsAttrName(), TypeAttr::get(ft));
      // Results
      s.addTypes(resultTypes);
      // Nested region
      auto *bodyPtr = s.addRegion();
      Region &body = *bodyPtr;
      OnClockOp::ensureTerminator(body, b, s.location);
  }
      

    ParseResult OnClockOp::parse(OpAsmParser &parser,
				 OperationState &result) {
      // parse clocks

      SmallVector<Type,4> inputFlags;
      SmallVector<Type,4> outputFlags;
      operands_vect inputClockOps;
      operands_vect outputClockOps;

      if (parser.parseLParen())
	return failure();

      if (ClockA::parseClockList(parser, inputClockOps, inputFlags))
	return failure();

      if (parser.parseColon())
	return failure();

      if (ClockA::parseClockList(parser, outputClockOps, outputFlags))
	return failure();

      if (parser.parseRParen())
	return failure();

      operands_vect clockOps;
      clockOps.append(inputClockOps);
      clockOps.append(outputClockOps);

      auto builder = parser.getBuilder();
      Type t = builder.getFunctionType(inputFlags, outputFlags);

      if (ClockA::resolveClockOps(parser,result,clockOps))
	return failure();

      result.addAttribute(getFlagsAttrName(), TypeAttr::get(t));
      
      // parse inside op

      auto *bodyPtr = result.addRegion();
      Region &body = *bodyPtr;
      OptionalParseResult opr = parser.parseOptionalRegion(body);
      if (opr.hasValue()) {
	Block &block = body.front();
	Operation &opInside = block.front();
	OnClockOp::ensureTerminator(body, builder, result.location);
	if(block.getOperations().size() != 2)
	  return failure();
	result.addTypes(opInside.getResultTypes());
      }
      return success();
  }
    
    void OnClockOp::print(OpAsmPrinter &p) {
      
      // PRINT CLOCK
      operand_iterator it = getOperation()->operand_begin();
      Type flagsType = getFlagsType();
      ClockA::printClock(p,it,flagsType,":");
      // PRINT THE CLOCKED OP
      p << "{ ";
      SmallVector<StringRef> defaultDialectStack{"builtin"};
      Operation *op = nested();
      if (auto *opInfo = op->getAbstractOperation())
	opInfo->printAssembly(op, p, defaultDialectStack.back());
      else
	p.printGenericOp(op);
      p << " }";
    }

    LogicalResult OnClockOp::verify() {
      return success();
    }
  }
}
