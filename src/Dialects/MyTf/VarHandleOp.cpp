#include "VarHandleOp.h"
#include "ResourceType.h"
#include <cstddef>

namespace mlir {
  namespace mytf {

    StringRef VarHandleOp::getSharedName() {
      Attribute attr = getOperation()->getAttrDictionary().get(getSharedNameKey());
	assert(attr != NULL);
	assert(attr.isa<StringAttr>());
	StringAttr stringAttr = attr.cast<StringAttr>();
	return stringAttr.getValue();
    }

    void VarHandleOp::build(OpBuilder &builder, OperationState &result,
			    StringRef sharedName, Type t) {
      result.addTypes({t});
      result.addAttribute(getSharedNameKey(),
			  StringAttr::get(builder.getContext(), sharedName));
    }

    ParseResult VarHandleOp::parse(OpAsmParser &parser,
				   OperationState &result) {
      Type t;
      if (parser.parseLParen()
	  || parser.parseRParen()
	  || parser.parseOptionalAttrDict(result.attributes)
	  || parser.parseColon()
	  || parser.parseLParen()
	  || parser.parseRParen()
	  || parser.parseArrow()
	  || parser.parseType(t))
	return failure();
      parser.addTypeToList(t, result.types);
      return success();
    }

    ArrayRef<StringRef> VarHandleOp::getAttributeNames() {
      static StringRef attrNames[] = { getSharedNameKey() };
      return makeArrayRef(attrNames);
    }

    LogicalResult VarHandleOp::verify() {
      if (getOperation()->getAttrDictionary().get(getSharedNameKey()) == NULL) {
	return emitOpError() << getSharedNameKey() << "attribute needed.";
      }
      if (!getResult().getType().isa<TensorType>()) {
      	return emitOpError() << "Type error : should be a tensor.";
      }
      return success();
    }

    void VarHandleOp::print(OpAsmPrinter &p) {
      p << "() ";
      p << getOperation()->getAttrDictionary();
      p << " : () -> ";
      p << getResult().getType();

    }
  }
}
