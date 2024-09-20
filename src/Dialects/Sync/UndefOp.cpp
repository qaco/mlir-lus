#include "UndefOp.h"

namespace mlir {
  namespace sync {

    void UndefOp::build(Builder &odsBuilder,
			OperationState &odsState,
			Type t) {
      odsState.addTypes(t);
    }

    ParseResult UndefOp::parse(OpAsmParser &parser, OperationState &result) {
      Type t;
      if (parser.parseColonType(t))
	return failure();
      parser.addTypeToList(t, result.types);
      return success();
    }

    void UndefOp::print(OpAsmPrinter &p) {
      p << " : " << getResult().getType();
    }

    LogicalResult UndefOp::verify() {
      return success();
    }

  }
}
