#include "../../Dialects/Lus/NodeTest.h"
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

    class ClassicClockCalculusPass : public PassWrapper<
      ClassicClockCalculusPass,
      OperationPass<NodeTestOp>> {
      void runOnOperation() override;
    };

    void ClassicClockCalculusPass::runOnOperation() {
      NodeTestOp nodeOp = getOperation();

      ClassicClocker clocker(nodeOp);
      ClassicClockEnv classicEnv = clocker.infer();
    }
    
    std::unique_ptr<Pass> createClassicClockCalculusPass() {
      return std::make_unique<ClassicClockCalculusPass>();
    }
  }
}
