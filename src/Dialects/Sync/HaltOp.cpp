#include "HaltOp.h"

namespace mlir {
  namespace sync {
    void HaltOp::build(Builder &builder, OperationState &state) {}
    ParseResult HaltOp::parse(OpAsmParser &parser, OperationState &result) {
	return success();
      }
    LogicalResult HaltOp::verify() { return success(); }
    void HaltOp::print(OpAsmPrinter &p) { }
  }
}
