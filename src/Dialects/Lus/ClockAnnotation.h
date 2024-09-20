//  -*- C++ -*- //

#ifndef MLIRLUS_CLOCK_ANNOTATION_H
#define MLIRLUS_CLOCK_ANNOTATION_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "ClockType.h"

namespace mlir {
  namespace lus {

    using operands_vect = SmallVector<OpAsmParser::OperandType,4>;
    using operand_range = OperandRange;
      using operand_iterator = operand_range::iterator;
    
    struct ClockA {
      static ParseResult parseClock(OpAsmParser &parser,
				    operands_vect &clockOps,
				    SmallVector<Type,4> &flagsTys);
      static ParseResult resolveClockOps(OpAsmParser &parser,
					 OperationState &result,
					 operands_vect &clockOps);
      static ParseResult parseClockList(OpAsmParser &parser,
					operands_vect &clockOps,
					SmallVector<Type,4> &flagsTys);
      static ParseResult parseFullClock(OpAsmParser &parser,
					operands_vect &clockOps,
					Type &flagsTy);
      static operand_iterator printClockAux(OpAsmPrinter &p,
					    operand_iterator it,
					    std::vector<bool> &flags);
      static void printClock(OpAsmPrinter &p,
			     operand_iterator it,
			     Type flagsType,
			     StringRef sep);
    };
  }
}

#endif
