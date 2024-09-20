#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "../../Dialects/Lus/NodeTest.h"
#include "../../Dialects/Lus/OutputOp.h"
#include "../../Dialects/Sync/OutputOp.h"
#include "../../Dialects/Sync/TickOp.h"
#include "../../Dialects/Sync/SyncOp.h"
#include "../../Dialects/Sync/InputOp.h"
#include "../../Dialects/Lus/Clocking/ClassicClocker.h"

namespace mlir {
  namespace lus {

    struct ExplicitSignalsPass : public PassWrapper< ExplicitSignalsPass,
						     OperationPass<NodeTestOp>> {
      void runOnOperation() override;
    };
      
    void ExplicitSignalsPass::runOnOperation() {

      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stdout", err);

      NodeTestOp nodeOp = getOperation();
      
	  // if (!nodeOp.outputsExplicit())
	  //   nodeOp.makeOutputsExplicit();

	  // Gen input signals
	  SmallVector<sync::InputOp,4> inputVect;
	  OpBuilder nb(&nodeOp.getBody());
	  for (Value a : nodeOp.getInputs()) {
	    sync::InputOp i = nb.create<sync::InputOp>(nodeOp.getLoc(), a, false);
	    a.replaceAllUsesExcept(i.getResult(), {i});
	    inputVect.push_back(i);
	  }

	  YieldOp y = nodeOp.getYield();
	  OpBuilder b(y);
	  Location loc = y.getLoc();

	  SmallVector<Value, 4> states;
	  states.append(y.getStates().begin(),
			y.getStates().end());
	  
	  // Gen output signals
	  SmallVector<Value,4> outputSyncs;
	  for (auto e: llvm::zip(nodeOp.getOutputs(), y.getOutputs())) {
	    Value oSig = get<0>(e);
	    Value oVal = get<1>(e);
	    sync::OutputOp o = b.create<sync::OutputOp>(loc, oSig, oVal);
	    outputSyncs.push_back(o.getResult());
	  }

	  // Gen tick
	  SmallVector<Value, 4> toTick;
	  toTick.append(outputSyncs.begin(),outputSyncs.end());
	  toTick.append(states.begin(),states.end());
	  sync::TickOp t = b.create<sync::TickOp>(loc, toTick);

	  // Gen sync
	  SmallVector<Value, 4> nstates;
	  if (states.size() > 0) {
	    sync::SyncOp s = b.create
	      <sync::SyncOp>(loc, t.getResult(), states);
	    nstates.append(s.getResults().begin(),s.getResults().end());
	  }

	  // Gen a new yield
	  SmallVector<Value, 4> outs;
	  outs.append(y.getOutputs().begin(),y.getOutputs().end());
	  b.create<YieldOp>(loc, nstates, outs);
		    
	  y.erase();
    }

    std::unique_ptr<Pass> createExplicitSignalsPass() {
      return std::make_unique<ExplicitSignalsPass>();
    }
    
  }
}
