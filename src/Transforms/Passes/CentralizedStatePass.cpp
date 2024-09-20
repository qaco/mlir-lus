#include "../../Dialects/Lus/NodeTest.h"
#include "../../Dialects/Lus/FbyOp.h"
#include "../../Dialects/Lus/InitOp.h"
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

    class CentralizedStatePass : public PassWrapper<
      CentralizedStatePass,
      OperationPass<NodeTestOp>> {
      void runOnOperation() override;
    };

    void CentralizedStatePass::runOnOperation() {
      NodeTestOp nodeOp = getOperation();
      ClassicClocker clocker(nodeOp);
      ClassicClockEnv classicEnv = clocker.infer();

      nodeOp.walk([&](FbyOp fbyOp) {

	  ClassicClock fbyClock = classicEnv.getClockOfOp(fbyOp);
	  assert(fbyClock.isBase());
	  assert(!isa<OnClockOp>(fbyOp.getOperation()->getParentOp()));

	  OpBuilder b(fbyOp);
	  Location loc = fbyOp.getLoc();

	  // Add an explicit state to the node
	  Value s = nodeOp.addState(b, fbyOp.getResult().getType());
	  fbyOp.getResult().replaceAllUsesWith(s);

	  // yield the explicit state at cycle k > 0
	  Value os = fbyOp.getRhs();
	  YieldOp oy = nodeOp.getYield();
	  SmallVector<Value, 4> states;
	  states.append(oy.getStates().begin(),
			oy.getStates().end());
	  states.push_back(os);
	  SmallVector<Value, 4> outputs;
	  outputs.append(oy.getOutputs().begin(),
			 oy.getOutputs().end());
	  oy.addState(os);

	  // initial value
	  InitOp initOp = b.create<InitOp>(loc,s,fbyOp.getLhs());
	  s.replaceAllUsesExcept(initOp.getResult(),{initOp});
	  fbyOp.erase();

	});
	 
    }
    
    std::unique_ptr<Pass> createCentralizedStatePass() {
      return std::make_unique<CentralizedStatePass>();
    }
  }
}
