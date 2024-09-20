#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
// #include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {

    struct LinalgCopyPass : public PassWrapper< LinalgCopyPass,
						 OperationPass<FuncOp>> {
      void runOnOperation() override;
    };
      
    void LinalgCopyPass::runOnOperation() {

      FuncOp funcOp = getOperation();

      funcOp.walk([&](linalg::CopyOp copyOp) {
	  copyOp.output().replaceAllUsesWith(copyOp.input());
	  copyOp.erase();
	});
      
    }

    std::unique_ptr<Pass> createLinalgCopyPass() {
      return std::make_unique<LinalgCopyPass>();
    }
}
