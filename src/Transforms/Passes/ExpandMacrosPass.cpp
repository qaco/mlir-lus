#include "../../Dialects/Lus/lus.h"
#include "../../Dialects/Lus/MergeOp.h"
#include "../../Dialects/Lus/WhenOp.h"
#include "../../Dialects/Lus/NodeTest.h"
#include "../../Dialects/Lus/InitOp.h"
#include "../../Dialects/Lus/KPeriodicOp.h"
#include "../Utilities/Helpers.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace std;

namespace mlir {
  namespace lus {

    class ExpandMacrosPass : public PassWrapper<
      ExpandMacrosPass,
      OperationPass<NodeTestOp>> {
      void runOnOperation() override;
    };

    void ExpandMacrosPass::runOnOperation() {
      ConversionTarget target(getContext());
      // target.addLegalDialect<pssa::Pssa,
      // 			     StandardOpsDialect
      // 			     >();
      // target.addIllegalDialect<Lus>();
      // target.addLegalOp <
      // 	WhenOp,
      // 	MergeOp,
      // 	NodeOp,
      // 	PreOp,
      // 	lus::YieldOp
      // 	>();

      NodeTestOp n = getOperation();
      n.walk([&](InitOp initOp) {
	  OpBuilder builder(initOp);
	  Location loc = initOp.getLoc();
	  Value stateVal = initOp.getOperand(0);
	  Value initVal = initOp.getOperand(1);
	  KperiodicOp kpOp = builder.create<KperiodicOp>(loc,"0(1)");
	  Cond<Value> condT(kpOp.getResult(),false);
	  Cond<Value> condF(kpOp.getResult(),true);
	  WhenOp whenTOp = builder.create<WhenOp>(loc,condT,stateVal);
	  WhenOp whenFOp = builder.create<WhenOp>(loc,condF,initVal);
	  MergeOp mergeOp = builder.create<MergeOp>(loc,condT,
						    whenTOp.getResult(),
						    whenFOp.getResult());
	  initOp.getResult().replaceAllUsesWith(mergeOp.getResult());
	  initOp.erase();
	});
    }

    std::unique_ptr<Pass> createExpandMacrosPass() {
      return std::make_unique<ExpandMacrosPass>();
    }
  }
}
