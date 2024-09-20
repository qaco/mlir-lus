#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
// #include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "../../Dialects/Lus/Node.h"
#include "../../Dialects/Pssa/pssa.h"
#include "../../Dialects/Pssa/OutputOp.h"
#include "../../Dialects/Sync/TickOp.h"
#include "../../Dialects/Sync/SyncOp.h"
#include "../../Dialects/Sync/InputOp.h"
#include "../../Dialects/Lus/ClockAnalysis.h"

namespace mlir {
  namespace lus {

    struct GenEnvPass : public PassWrapper< GenEnvPass,
						 OperationPass<NodeOp>> {
      void runOnOperation() override;
    };
      
    void GenEnvPass::runOnOperation() {

      NodeOp nodeOp = getOperation();

      if (!nodeOp.hasBody())
      	return;

      // Gen input signals
      SmallVector<sync::InputOp,4> inputVect;
      OpBuilder nb(&nodeOp.getBody());
      for (Value a : nodeOp.getInputs()) {
	sync::InputOp i = nb.create<sync::InputOp>(nodeOp.getLoc(), a, false);
	a.replaceAllUsesExcept(i.getResult(), {i});
	inputVect.push_back(i);
      }

      // Ensure dominance when one input signal clocks another
      ClockAnalysis analysis(nodeOp);
      analysis.analyse();
      ClockTree *clockTree = &analysis.getClockTree();
      int domWrong;
      do {
	domWrong = 0;
	for (sync::InputOp i: inputVect) {
	  for (ClockTree::Edge e: clockTree->path(i)) {
	    Value clock = (e.getData());
	    if (auto defClock = clock.getDefiningOp()) {
	      if (i.getOperation()->isBeforeInBlock(defClock)) {
		domWrong = 1;
		defClock->moveBefore(i);
	      }
	    }
	  }
	}
      } while (domWrong);

      
      
      // Extend lus::YieldOp
      nodeOp.walk([&](YieldOp y) {
	  OpBuilder b(y);
	  Location loc = y.getLoc();

	  // Gen output signals
	  int64_t i = 0;
	  SmallVector<Value,4> outputSyncs;
	  for (Value v: y.getOutputs()) {
	    pssa::OutputOp o = b.create<pssa::OutputOp>(loc, i, v);
	    outputSyncs.push_back(o.getResult());
	    i++;
	  }

	  // Gen tick
	  sync::TickOp t = b.create<sync::TickOp>(loc, outputSyncs);

	  // Gen sync
	  SmallVector<Value, 4> states;
	  states.append(y.getStates().begin(),
			y.getStates().end());
	  sync::SyncOp s = b.create
	    <sync::SyncOp>(loc, t.getResult(), states);

	  // Gen a new yield
	  SmallVector<Value, 4> nstates;
	  nstates.append(s.getValues().begin(),s.getValues().end());
	  SmallVector<Value, 4> outs;
	  outs.append(y.getOutputs().begin(),y.getOutputs().end());
	  b.create<YieldOp>(loc, nstates, outs);
		    
	  y.erase();
	});
    }

    std::unique_ptr<Pass> createGenEnvPass() {
      return std::make_unique<GenEnvPass>();
    }
    
  }
}
