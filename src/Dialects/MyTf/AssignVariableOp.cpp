#include "AssignVariableOp.h"
#include "ResourceType.h"

namespace mlir {
  namespace mytf {

    void AssignVariableOp::build(Builder &builder,
				 OperationState &state,
				 Value resource, Value value) {
      state.addOperands({resource, value});
    }

    ParseResult AssignVariableOp::parse(OpAsmParser &parser,
					OperationState &result) {
      OpAsmParser::OperandType op1;
      Type op1Type;
      OpAsmParser::OperandType op2;
      Type op2Type;
      if (parser.parseLParen()
	  || parser.parseOperand(op1)
	  || parser.parseComma()
	  || parser.parseOperand(op2)
	  || parser.parseRParen()
	  || parser.parseOptionalAttrDict(result.attributes)
	  || parser.parseColon()
	  || parser.parseLParen()
	  || parser.parseType(op1Type)
	  || parser.parseComma()
	  || parser.parseType(op2Type)
	  || parser.parseRParen()
	  || parser.parseArrow()
	  || parser.parseLParen()
	  || parser.parseRParen()
	  || parser.resolveOperand(op1, op1Type, result.operands)
	  || parser.resolveOperand(op2, op2Type, result.operands))
	return failure();
      return success();
    }

    void AssignVariableOp::print(OpAsmPrinter &p) {
      p << getOperationName()
	<< "(" << getOperand(0) << ", " << getOperand(1) << ") "
	<< getOperation()->getAttrDictionary()
	<< " : (" << getOperand(0).getType()
	<< ", " << getOperand(1).getType() << ") -> ()";
    }

    LogicalResult AssignVariableOp::verify() {
      Type op1T = getOperand(0).getType();
      Type op2T = getOperand(1).getType();

      ResourceType resourceT = ResourceType::get(getContext(),
						 op2T);
      RankedTensorType tensorT = RankedTensorType::get({},
				     resourceT);

      if (tensorT != op1T)
	return emitOpError() << "Type error.";

      return success();
    }

  }
}
