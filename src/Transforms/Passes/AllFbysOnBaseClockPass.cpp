#include "../../Dialects/Lus/NodeTest.h"
#include "../../Dialects/Lus/FbyOp.h"
#include "../../Dialects/Lus/Clocking/ClassicClocker.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "../LusToLus/SortAlongClocks.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace std;
using namespace mlir::arith;
using namespace mlir::tensor;

namespace mlir {

  namespace lus {

    class AllFbysOnBaseClockPass : public PassWrapper<
      AllFbysOnBaseClockPass,
      OperationPass<NodeTestOp>> {
      void runOnOperation() override;
    };

    void AllFbysOnBaseClockPass::runOnOperation() {
      NodeTestOp nodeOp = getOperation();
      ClassicClocker clocker(nodeOp);
      ClassicClockEnv classicEnv = clocker.infer();

      nodeOp.walk([&](FbyOp fbyOp) {

	  ClassicClock fbyClock = classicEnv.getClockOfOp(fbyOp);

	  Location loc = fbyOp.getLoc();
	  OpBuilder builder(fbyOp);
	  if (auto onClockOp = dyn_cast<OnClockOp>(fbyOp.getOperation()->getParentOp())) {
	    builder.setInsertionPoint(onClockOp);
	  }

	  SmallVectorImpl<Subsampling>& subsamplings = fbyClock.getSubsamplings();

	  for (int i = subsamplings.size() - 1; i >= 0; i--) {
	    Subsampling ss = subsamplings[i];

	    Cond<Value> cond(ss.data, !ss.on_or_onnot);
	    Cond<Value> condNot(ss.data, ss.on_or_onnot);

	    // Replace the current result state by the sampled version
	    // of itself
	    WhenOp when1Op = builder.create<WhenOp>(loc,cond,
						    fbyOp.getResult());
	    fbyOp.getResult().replaceAllUsesExcept(when1Op.getResult(),
						   {when1Op});

	    // Transmission of the old state (if c = false) or
	    // the next state
	    WhenOp when2Op = builder.create<WhenOp>(loc,condNot,
						    fbyOp.getResult());
	    Value rhsFby = fbyOp.getRhs();
	    MergeOp mergeOp = builder.create<MergeOp>(loc,cond,
						      rhsFby,
						      when2Op.getResult());
	    fbyOp.getOperation()->setOperand(1,mergeOp.getResult());
	  }

	  if (auto onClockOp = dyn_cast<OnClockOp>(fbyOp.getOperation()->getParentOp())) {
	    Operation* clonedFby = builder.clone(*fbyOp.getOperation());
	    onClockOp.getOperation()->replaceAllUsesWith(clonedFby);
	    onClockOp.erase();
	  }
	});
	 
    }
    
    std::unique_ptr<Pass> createAllFbysOnBaseClockPass() {
      return std::make_unique<AllFbysOnBaseClockPass>();
    }
  }
}
