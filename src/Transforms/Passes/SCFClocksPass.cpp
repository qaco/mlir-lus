#include "../../Dialects/Lus/NodeTest.h"
#include "../../Dialects/Lus/OnClockOp.h"
#include "../../Dialects/Sync/UndefOp.h"
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

    class SCFClocksPass : public PassWrapper<
      SCFClocksPass,
      OperationPass<NodeTestOp>> {
      void runOnOperation() override;
    };

    void SCFClocksPass::runOnOperation() {

      NodeTestOp nodeOp = getOperation();

      nodeOp.walk([&](OnClockOp onClockOp) {
	  Location loc = onClockOp.getLoc();
	  ClassicClock cc = onClockOp.eqClock();
	  Operation *nested = onClockOp.nested();
	  
	  if (cc.isBase()) {
	    OpBuilder b(onClockOp);
	    Operation* cloned = b.clone(*nested);
	    onClockOp.replaceAllUsesWith(cloned);
	    onClockOp.erase();
	    return;
	  }

	  OpBuilder ifB(onClockOp);
	  SmallVectorImpl<Subsampling> &ss = cc.getSubsamplings();
	  Operation* topLevelOp;
	  for (unsigned i = 0; i < ss.size(); i++) {
	    Subsampling s = ss[i];

	    // If creation
	    
	    scf::IfOp ifOp = ifB.create<scf::IfOp>(loc,
						   onClockOp.getResultTypes(),
						   s.data, true);
	    if (i == 0) {
	      topLevelOp = ifOp.getOperation();
	    }
	    else {
	      OperationState y1State(loc, scf::YieldOp::getOperationName());
	      scf::YieldOp::build(ifB,y1State,ifOp->getResults());
	      ifB.createOperation(y1State);
	    }

	    // Builders management

	    ifB = ifOp.getThenBodyBuilder();
	    OpBuilder uB = ifOp.getElseBodyBuilder();
	    if (!s.on_or_onnot) {
	      ifB = ifOp.getElseBodyBuilder();
	      uB = ifOp.getThenBodyBuilder();
	    }

	    // The undef branch
	    
	    SmallVector<Value,4> myUndefs;
	    for (Type t: onClockOp.getResultTypes()) {
	      sync::UndefOp uOp = uB.create<sync::UndefOp>(loc,t);
	      myUndefs.push_back(uOp.getResult());
	    }
	    OperationState y2State(loc, scf::YieldOp::getOperationName());
	    scf::YieldOp::build(uB,y2State,myUndefs);
	    uB.createOperation(y2State);

	    // The def branch

	    if (i == ss.size() - 1) {
	      Operation* cloned = ifB.clone(*nested);
	      OperationState y1State(loc, scf::YieldOp::getOperationName());
	      scf::YieldOp::build(ifB,y1State,cloned->getResults());
	      ifB.createOperation(y1State);
	    }
	  }
	  
	  // Finalization

	  onClockOp.replaceAllUsesWith(topLevelOp);
	  onClockOp.erase();
	  
	});

      nodeOp.walk([&](WhenOp whenOp) {
	  whenOp.getResult().replaceAllUsesWith(whenOp.getDataInput());
	  whenOp.erase();
	});

      nodeOp.walk([&](MergeOp m) {
	  OpBuilder mb(m);
	  Location ml = m.getLoc();
	  
	  SelectOp s = mb.create<SelectOp>(ml, m.getCondValue(),
					   m.getTrueInput(),
					   m.getFalseInput());
	  m.getResult().replaceAllUsesWith(s.getResult());
	  m.erase();
	});
    }
    
    std::unique_ptr<Pass> createSCFClocksPass() {
      return std::make_unique<SCFClocksPass>();
    }
  }
}
