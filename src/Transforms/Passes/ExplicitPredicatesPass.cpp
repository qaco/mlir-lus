#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "../../Dialects/Lus/NodeTest.h"
#include "../../Dialects/Lus/Clocking/ClassicClocker.h"

using namespace mlir;
using namespace std;

namespace mlir {

  namespace lus {

    class ExplicitPredicatesPass : public PassWrapper<
      ExplicitPredicatesPass,
      OperationPass<NodeTestOp>> {
      void runOnOperation() override;
    };

    // void ExplicitPredicatesPass::runOnOperation() {
            
    //   NodeTestOp nodeOp = getOperation();
    //   ClassicClocker clocker(nodeOp);
    //   ClassicClockEnv classicEnv = clocker.infer();

    //   SmallVector<Operation*,4> ops;
    //   for (Operation &op: nodeOp.getBody().front()) {
    //   	if (!isa<OnClockOp>(op) && !isa<YieldOp>(op))
    //   	  ops.push_back(&op);
    //   }

    //   for (auto it = ops.end(); it > ops.begin(); it--) {

    //   	Operation *op = *(it-1);
    //   	OpBuilder b(op);

    //   	ClassicClock eqCc = classicEnv.getClockOfOp(op);
	
    //   	SmallVector<ClassicClock,4> outputClocks;
    //   	for (Value v: op->getResults()) {
    //   	  ClassicClock cc = classicEnv.getClockOfValue(v);
    //   	  outputClocks.push_back(cc);
    //   	}

    //   	OnClockOp clocked = b.create<OnClockOp>(op->getLoc(),
    //   						eqCc,
    //   						outputClocks,
    //   						op);

    //   	b.setInsertionPoint(&clocked.getBody().front().front());
    //   	b.clone(*op);

    //   	for (auto e: llvm::zip(op->getResults(),
    //   			       clocked.getResults())) {
    //   	  Value oldVal = get<0>(e);
    //   	  ClassicClock cc = classicEnv.getClockOfValue(oldVal);
    //   	  Value newVal = get<1>(e);
    //   	  oldVal.replaceAllUsesWith(newVal);
    //   	  classicEnv.bindValue(newVal,cc);
    //   	}
    //   }

    //   for (Operation *op: ops) {
    //   	op->erase();
    //   }
    // }

    void ExplicitPredicatesPass::runOnOperation() {

      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stdout", err);


      NodeTestOp nodeOp = getOperation();

      ClassicClocker clocker(nodeOp);
      ClassicClockEnv classicEnv = clocker.infer();

      llvm::SmallMapVector<Operation*,Operation*,4> replacements;

      nodeOp.walk([&](Operation* op) {
	  if (isa<OnClockOp>(op)
	      || isa<OnClockNodeOp>(op)
	      || isa<YieldOp>(op)
	      || isa<OnClockOp>(op->getParentOp())
	      || isa<NodeTestOp>(op)) {
	    return;
	  }

	  OpBuilder b(op);
	      
	  ClassicClock eqCc = classicEnv.getClockOfOp(op);

	  SmallVector<ClassicClock,4> outputClocks;
	  for (Value v: op->getResults()) {
	    ClassicClock cc = classicEnv.getClockOfValue(v);
	    outputClocks.push_back(cc);
	  }
	      
	  OnClockOp clocked = b.create<OnClockOp>(op->getLoc(),
						  eqCc,
						  outputClocks,
						  op->getResultTypes());
	  b.setInsertionPoint(&clocked.getBody().front().front());
	  b.clone(*op);
	
	  replacements[op] = clocked.getOperation();
	});

      for (auto p: replacements) {
      	for (auto e: llvm::zip(p.first->getResults(),
      			       p.second->getResults()))
      	  get<0>(e).replaceAllUsesWith(get<1>(e));
      	p.first->erase();
      }
    }
    
    std::unique_ptr<Pass> createExplicitPredicatesPass() {
      return std::make_unique<ExplicitPredicatesPass>();
    }
  }
}
