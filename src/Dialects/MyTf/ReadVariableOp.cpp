#include "ReadVariableOp.h"
#include "ResourceType.h"

namespace mlir {
  namespace mytf {

    void ReadVariableOp::build(Builder &builder,
			       OperationState &state,
			       Value resource) {
      Type t = resource.getType();
      assert(t.isa<TensorType>());
      TensorType tt = t.dyn_cast<TensorType>();
      ResourceType rt = tt.getElementType().cast<ResourceType>();
      Type it = rt.getType();
      state.addOperands(resource);
      state.addTypes(it);
    }

    ParseResult ReadVariableOp::parse(OpAsmParser &parser,
				      OperationState &result) {
      OpAsmParser::OperandType op;
      Type opType;
      Type resType;
      if (parser.parseLParen()
	  || parser.parseOperand(op)
	  || parser.parseRParen()
	  || parser.parseOptionalAttrDict(result.attributes)
	  || parser.parseColon()
	  || parser.parseType(opType)
	  || parser.resolveOperand(op, opType, result.operands)
	  || parser.parseArrow()
	  || parser.parseType(resType))
	return failure();
      result.addTypes({resType});
      return success();
    }

    void ReadVariableOp::print(OpAsmPrinter &p) {
      p << getOperationName() << "(" << getOperand() << ") "
	<< getOperation()->getAttrDictionary()
	<< " : (" << getOperand().getType() << ")"
	<< " -> " << getResult().getType();
    }

    LogicalResult ReadVariableOp::verify() {
      Type opT = getOperand().getType();
      Type resT = getResult().getType();

      ResourceType resourceT = ResourceType::get(getContext(),
					 resT);
      RankedTensorType tensorT = RankedTensorType::get({},
				     resourceT);
      if (tensorT != opT)
	return emitOpError() << "Type error.";

      return success();
    }
  }
}
