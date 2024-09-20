#include "../../Dialects/Lus/lus.h"
#include "../../Dialects/Pssa/pssa.h"
#include "../../Dialects/Lus/WhenOp.h"
#include "../../Dialects/Lus/MergeOp.h"
#include "../LusToLus/SortAlongClocks.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace std;

namespace mlir {
  namespace lus {

    class SortAlongClocksPass : public PassWrapper<
      SortAlongClocksPass,
      OperationPass<NodeOp>> {
      void runOnOperation() override;
    };

    void SortAlongClocksPass::runOnOperation() {
      ConversionTarget target(getContext());
      target.addLegalDialect<pssa::Pssa,
			     StandardOpsDialect
			     >();
      target.addIllegalDialect<Lus>();
      target.addLegalOp <
	WhenOp,
	MergeOp,
	NodeOp,
	lus::YieldOp
	>();

      Operation* op = getOperation();
      NodeOp nodeOp = dyn_cast<NodeOp>(op);
      if (!nodeOp.hasBody())
	return;
      SortAlongClocks sortAlongClocks;
      sortAlongClocks(nodeOp);
    }

    std::unique_ptr<Pass> createSortAlongClocksPass() {
      return std::make_unique<SortAlongClocksPass>();
    }
  }
}
