#include "ClockAnnotation.h"

namespace mlir {
  namespace lus {

    ParseResult ClockA::parseClock(OpAsmParser &parser,
				   operands_vect &clockOps,
				   SmallVector<Type,4> &flagsTys) {
      auto builder = parser.getBuilder();
      std::vector<bool> flags;
      
      if (parser.parseKeyword("base"))
	return failure();

      while (succeeded(parser.parseOptionalKeyword("on"))) {
	bool flag = true;
	if (succeeded(parser.parseOptionalKeyword("not")))
	  flag = false;
	flags.push_back(flag);
	OpAsmParser::OperandType clock;
	if (parser.parseOperand(clock))
	  return failure();
	clockOps.push_back(clock);
      }
      Type t = builder.getType<ClockType,std::vector<bool>>(flags);
      flagsTys.push_back(t);
	
      return success();
    }

    ParseResult ClockA::resolveClockOps(OpAsmParser &parser,
					OperationState &result,
					operands_vect &clockOps) {
      auto builder = parser.getBuilder();
      Type bty = IntegerType::get(builder.getContext(),1);
      for (auto v: clockOps) {
	if (parser.resolveOperand(v, bty,
				  result.operands))
	  return failure();
      }
      return success();
    }
    
    ParseResult ClockA::parseClockList(OpAsmParser &parser,
				       operands_vect &clockOps,
				       SmallVector<Type,4> &flagsTys) {
      if (parser.parseLParen())
	return failure();
      if (parser.parseOptionalRParen()) {
	do {
	  if (parseClock(parser,clockOps, flagsTys))
	    return failure();
	} while (succeeded(parser.parseOptionalComma()));
	if (parser.parseRParen())
	  return failure();
      }
	return success();
    }

    ParseResult ClockA::parseFullClock(OpAsmParser &parser,
				       operands_vect &clockOps,
				       Type &flagsTy) {
      auto builder = parser.getBuilder();
      SmallVector<Type,4> inputFlags;
      SmallVector<Type,4> outputFlags;
      operands_vect inputClockOps;
      operands_vect outputClockOps;
      
      if (parser.parseLParen())
	return failure();

      if (ClockA::parseClockList(parser, inputClockOps, inputFlags))
	  return failure();
	clockOps.append(inputClockOps);

      if (parser.parseArrow())
	return failure();

      if (ClockA::parseClockList(parser, outputClockOps, outputFlags))
	  return failure();
	clockOps.append(outputClockOps);

      flagsTy = builder.getFunctionType(inputFlags, outputFlags);

      if (parser.parseRParen())
	return failure();
      return success();
    }

    operand_iterator ClockA::printClockAux(OpAsmPrinter &p,
					   operand_iterator it,
					   std::vector<bool> &flags) {
      p << "base";
      size_t i = 0;
      for (bool flag: flags) {
	Value operand = *(it++);
	p << " on";
      	if (!flag)
      	  p << " not";

      	p << " " << operand;
	
      	if (i < flags.size() - 1)
      	  i++;
      }
      return it;
    }

    void ClockA::printClock(OpAsmPrinter &p,
			    operand_iterator it,
			    Type flagsType,
			    StringRef sep) {
      if (flagsType.isa<FunctionType>()) {
	p << " ((";
	FunctionType funcType = flagsType.cast<FunctionType>();
	// Print inputs
	size_t iIn = 0;
	for (auto ty: funcType.getInputs()) {
	  ClockType sTy = ty.cast<ClockType>();
	  std::vector<bool> flags = sTy.getSeq();
	  it = printClockAux(p,it,flags);
	  if (iIn < funcType.getNumInputs() -1) {
	    iIn++;
	    p << ", ";
	  }
	}
	// Print outputs
	p << ") " << sep << " (";
	size_t iOut = 0;
	for (auto ty: funcType.getResults()) {
	  ClockType sTy = ty.cast<ClockType>();
	  std::vector<bool> flags = sTy.getSeq();
	  it = printClockAux(p,it,flags);
	  if (iOut < funcType.getNumResults() -1) {
	    iOut++;
	    p << ", ";
	  }
	}
	p << ")) ";
      }
      else {
	assert(false);
      }
    }
  }
}
