#include "../../Dialects/Lus/NodeTest.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "../LusToLus/SortAlongClocks.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "../../Dialects/Lus/Clocking/ClassicClocker.h"

using namespace mlir;
using namespace std;
using namespace mlir::arith;
using namespace mlir::tensor;

namespace mlir {

  namespace lus {
    
    class RecomputeOrderPass : public PassWrapper<
      RecomputeOrderPass,
      OperationPass<NodeTestOp>> {
      void runOnOperation() override;
    };

    void RecomputeOrderPass::runOnOperation() {

      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stdout", err);

      NodeTestOp nodeOp = getOperation();

      ClassicClocker clocker(nodeOp);
      ClassicClockEnv classicEnv = clocker.infer();

      std::unordered_map<Operation*,llvm::SetVector<Value>> todo;
      llvm::SetVector<Value> defined;
      SmallVector<Operation*> order;

      // Init the defined values
      
      for (Value v: nodeOp.getInputs()) {
	defined.insert(v);
      }
      if (nodeOp.outputsExplicit()) {
	for (Value v: nodeOp.getOutputs()) {
	  defined.insert(v);
	}
      }
      for (Value v: nodeOp.getState()) {
	defined.insert(v);
      }

      // Fill the todo map
      
      for (Operation &op: nodeOp.getBody().front()) {

	if (isa<YieldOp>(&op))
	  continue;
	
	llvm::SetVector<Value> deps;

	// Operands and operands clocks give dependencies
	for (Value v: op.getOperands()) {
	  ClassicClock cc = classicEnv.getClockOfValue(v);
	  SmallVectorImpl<Subsampling> &ss = cc.getSubsamplings();
	  for (Subsampling s: ss) {
	    deps.insert(s.data);
	  }
	  deps.insert(v);
	}
	// Inside the clock annotation
	if (auto onClockOp = dyn_cast<OnClockOp>(op)) {
	  for (Value v: onClockOp.nested()->getOperands()) {
	    ClassicClock cc = classicEnv.getClockOfValue(v);
	    SmallVectorImpl<Subsampling> &ss = cc.getSubsamplings();
	    for (Subsampling s: ss) {
	      deps.insert(s.data);
	    }
	    deps.insert(v);
	  }
	}

	// Operation clock give dependencies
	ClassicClock cc = classicEnv.getClockOfOp(&op);
	SmallVectorImpl<Subsampling> &ss = cc.getSubsamplings();
	for (Subsampling s: ss) {
	  deps.insert(s.data);
	}
	
	todo[&op] = deps;
      }

      // Main algorithm

      bool fixed_point = false;
      // YieldOp yieldOp = nodeOp.getYield();
      
      while (!fixed_point) {
	
	unsigned todo_size = todo.size();
	auto it = todo.begin();

	// Iterate on (op, deps)
	while (it != todo.end()) {
	  auto pair = *it;
	  Operation *op = pair.first;
	  llvm::SetVector<Value> deps = pair.second;

	  // Check if deps contained in defined (operation respects SSA)
	  bool respectsSSA = true;
	  for (Value dep: deps) {
	    if (!defined.contains(dep)) respectsSSA = false;
	  }
	  
	  if (respectsSSA) {
	    order.push_back(op);
	    // op->moveBefore(yieldOp);
	    for (Value v: op->getResults())
	      defined.insert(v);
	    it = todo.erase(it);
	  }
	  else {
	    it++;
	  }
	}
	fixed_point = (todo_size == todo.size());

      }
      
      assert(todo.size() == 0);

      // Write

      OpBuilder b(nodeOp.getYield());

      for (Operation *op: order) {
	Operation *cloned = b.clone(*op);
	op->replaceAllUsesWith(cloned);
	op->erase();
      }

      // Turn on the dominance flag
      
      nodeOp.forceDominance();
      
    }

    std::unique_ptr<Pass> createRecomputeOrderPass() {
      return std::make_unique<RecomputeOrderPass>();
    }
  }
}
