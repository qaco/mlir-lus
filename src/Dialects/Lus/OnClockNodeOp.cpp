#include "OnClockNodeOp.h"
#include "Yield.h"
#include "llvm/ADT/StringExtras.h"
#include "ClockAnnotation.h"

namespace mlir {
  namespace lus {

    using operand_range = OperandRange;
    using operand_iterator = operand_range::iterator;

    // // TODO : factoriser cette fonction et la suivante
    // void OnClockNodeOp::clockedInputs(SmallVectorImpl<ClassicClock> &inputs) {
    //   FunctionType ft = getFlagsType().dyn_cast<FunctionType>();
    //   operand_iterator it = getOperation()->operand_begin();

    //   for (auto ty : ft.getInputs()) {
    // 	SmallVector<Subsampling,4> subsamplings;
    // 	ClockType cty = ty.cast<ClockType>();
    // 	std::vector<bool> flags = cty.getSeq();
    // 	for (bool flag: flags) {
    // 	  Value operand = *(it++);
    // 	  Subsampling ss(flag,operand);
    // 	  subsamplings.push_back(ss);
    // 	}
    // 	ClassicClock cc(ClassicClock::PREFIX_BASE,subsamplings);
    // 	inputs.push_back(cc);
    //   }
    // }

    // TODO : factoriser cette fonction et la précédente
    void OnClockNodeOp::clockedOutputs(SmallVectorImpl<ClassicClock> &outputs) {
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

    void OnClockNodeOp::build(Builder &b, OperationState &s,
			  SmallVectorImpl<ClassicClock>& inputClocks,
			  SmallVectorImpl<ClassicClock>& outputClocks) {
      // Input clocks
      SmallVector<Type,4> inputFlags;
      for (ClassicClock cc: inputClocks) {
	std::vector<bool> flags;
	for (auto ss: cc.getSubsamplings()) {
	  s.addOperands({ss.data});
	  flags.push_back(ss.on_or_onnot);
	}
	Type t = b.getType<ClockType,std::vector<bool>>(flags);
	inputFlags.push_back(t);
      }
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
      Type ft = b.getFunctionType(inputFlags,outputFlags);
      s.addAttribute(getFlagsAttrName(), TypeAttr::get(ft));
    }

    ParseResult OnClockNodeOp::parse(OpAsmParser &parser,
				 OperationState &result) {

      SmallVector<Type,4> inputFlags;
      SmallVector<Type,4> outputFlags;
      operands_vect inputClockOps;
      operands_vect outputClockOps;

      if (parser.parseLParen())
	return failure();

      if (ClockA::parseClockList(parser, inputClockOps, inputFlags))
	return failure();

      if (parser.parseArrow())
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
      
      return success();
  }
    
    void OnClockNodeOp::print(OpAsmPrinter &p) {
      
      operand_iterator it = getOperation()->operand_begin();
      Type flagsType = getFlagsType();
      ClockA::printClock(p,it,flagsType,"->");
      
    }

    LogicalResult OnClockNodeOp::verify() {
      return success();
    }
  }
}
