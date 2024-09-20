// -*- C++ -*- //

#ifndef CLASSIC_CLOCKER_H
#define CLASSIC_CLOCKER_H

#include "ClassicClock.h"
#include "../NodeTest.h"
#include "../OnClockOp.h"
#include "../WhenOp.h"
#include "../InstanceTest.h"
#include "../MergeOp.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace lus {

    class ClockedOutputsHandler {
    private:
      iterator_range<Block::args_iterator> abst;
      Operation::operand_range conc;
    public:
      ClockedOutputsHandler() = delete;
      ClockedOutputsHandler(iterator_range<Block::args_iterator> a,
			    Operation::operand_range c): abst(a),conc(c) {}
      Value concretize(Value a);
    };
    
    class ClassicClocker {
      
    private:

      static const bool DEBUG_MODE = false;
      ClassicClockEnv classicEnv;
      NodeTestOp node;
      
      void bindSignature();
      void bindOperation(Operation *op);
      void bindBody();
      bool unifyClocks(ClassicClock cc1, ClassicClock cc2);
      bool verboseUnifyClocks(ClassicClock cc1, ClassicClock cc2);
      ClassicClock incrementClock(ClassicClock initial,
				  bool whenotFlag,
				  Value cond);
      bool inferOp(Operation* op);
      bool verboseInferOp(Operation* op);
      bool inferWhen(WhenOp when);
      bool inferMerge(MergeOp merge);
      bool inferOnClock(OnClockOp onClock);
      bool inferInst(InstanceTestOp instanceOp);
      bool inferSimple(Operation* op);
      
    public:
      
      ClassicClocker (NodeTestOp n): node(n) {}
      ClassicClockEnv infer();
    };
    
  }
}

#endif
