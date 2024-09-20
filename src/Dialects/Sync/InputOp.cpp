#include "InputOp.h"
#include "SignalTypes.h"

namespace mlir {
  namespace sync {

    void InputOp::build(Builder &odsBuilder, OperationState &odsState,
			Value v, bool isSignal) {
      Type ity = IntegerType::get(odsBuilder.getContext(),64);
      odsState.addOperands(v);
      if (isSignal) {
	assert(v.getType().isa<SiginType>());
	SiginType st = v.getType().cast<SiginType>();
	odsState.addTypes(st.getType());
	Attribute issig = IntegerAttr::get(ity,isSigValue());
	odsState.addAttribute(getSigAttrName(),issig);
      }
      else {
	odsState.addTypes(v.getType());
	Attribute issig = IntegerAttr::get(ity,!isSigValue());
	odsState.addAttribute(getSigAttrName(),issig);
      }
    }

    ParseResult InputOp::parse(OpAsmParser &parser, OperationState &result) {
      auto builder = parser.getBuilder();
      Type t;
      OpAsmParser::OperandType op;
      if (parser.parseLParen()
	  || parser.parseOperand(op)
	  || parser.parseRParen()
	  || parser.parseColonType(t))
	return failure();
      SiginType st = builder.getType < SiginType, Type > (t);
      if (parser.resolveOperand(op, st, result.operands))
	return failure();
      parser.addTypeToList(t, result.types);
      return success();
    }

    void InputOp::print(OpAsmPrinter &p) {
      p << " ("
	<< getOperand() << ")";
      p << " : " << getResult().getType();
    }

    LogicalResult InputOp::verify() {
      if (isSig()) {
	Type ot = getOperand().getType();
	if (!ot.isa<SiginType>())
	  return emitOpError() << getOperationName()
			       << "must take a sigin type";
	
	SiginType st = ot.cast<SiginType>();
	if (getResult().getType() != st.getType())
	  return emitOpError() << "return type must be " << st.getType();
      }
      return success();
    }
  }
}
