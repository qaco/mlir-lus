#include "InitOp.h"

namespace mlir {
  namespace lus {
    void InitOp::build(Builder &odsBuilder,OperationState &odsState,
		       Value v, Value i) {
      odsState.addOperands(v);
      odsState.addOperands(i);
      odsState.addTypes(v.getType());
    }

    ParseResult InitOp::parse(OpAsmParser &parser, OperationState &result) {
      OpAsmParser::OperandType opS;
      OpAsmParser::OperandType opI;
      Type ty;
      if (parser.parseLParen() ||
	  parser.parseOperand(opS) ||
	  parser.parseComma() ||
	  parser.parseOperand(opI) ||
	  parser.parseRParen() ||
	  parser.parseColonType(ty) ||
	  parser.resolveOperand(opI, ty, result.operands) ||
	  parser.resolveOperand(opS, ty, result.operands))
	return failure();
      result.addTypes(ty);
      return success();
    }
    
    void InitOp::print(OpAsmPrinter &p) {
      p << "(";
      p.printOperands(getOperands());
      p << ") : " << getResult().getType();
    }
    
    LogicalResult InitOp::verify() {
      return success();
    }
  }
}
