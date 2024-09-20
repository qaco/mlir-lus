#include "../../Dialects/Lus/NodeTest.h"
#include "../../Dialects/Lus/InstanceTest.h"
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

    class InlineInstancesPass : public PassWrapper<
      InlineInstancesPass,
      OperationPass<NodeTestOp>> {
      void runOnOperation() override;
    };

    void InlineInstancesPass::runOnOperation() {
      NodeTestOp nodeOp = getOperation();
      OpBuilder topB(nodeOp);

      nodeOp.walk([&](InstanceTestOp instOp) {
	  // TODO
	  assert(!isa<OnClockOp>(instOp.getOperation()->getParentOp()));
	  OpBuilder instB(instOp);
	  NodeTestOp callee = instOp.getCalleeNode();
	  Operation *clonedCalleePtr = topB.clone(*callee.getOperation());
	  NodeTestOp clonedCallee = dyn_cast<NodeTestOp>(clonedCalleePtr);
	  YieldOp clonedCalleeYield = clonedCallee.getYield();
	  for (auto e: llvm::zip(clonedCallee.getInputs(),
				 instOp.getArgOperands())) {
	    get<0>(e).replaceAllUsesWith(get<1>(e));
	  }
	  for (Operation &op: clonedCallee.getBody().front()) {
	    if (isa<YieldOp>(&op))
	      continue;
	    Operation *clonedOp = instB.clone(op);
	    op.replaceAllUsesWith(clonedOp);
	  }
	  for (auto e: llvm::zip(instOp.getResults(),
				 clonedCalleeYield.getOperands())) {
	    get<0>(e).replaceAllUsesWith(get<1>(e));
	  }
	  clonedCalleePtr->erase();
	  instOp.erase();
	});
	 
    }
    
    std::unique_ptr<Pass> createInlineInstancesPass() {
      return std::make_unique<InlineInstancesPass>();
    }
  }
}
