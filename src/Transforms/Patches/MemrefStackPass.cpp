#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

    struct MemrefStackPass : public PassWrapper< MemrefStackPass,
						 OperationPass<FuncOp>> {
      void runOnOperation() override;
    };
      
    void MemrefStackPass::runOnOperation() {

      FuncOp funcOp = getOperation();

      funcOp.walk([&](memref::AllocOp alloc) {
	  Operation *allocOp = alloc.getOperation();
	  OpBuilder builder(allocOp);
	  Operation *alloca = builder.create<memref::AllocaOp>(alloc.getLoc(),
							       alloc.getType(),
		       				       allocOp->getOperands());
	  alloc.replaceAllUsesWith(alloca);
	  alloc.erase();
	});
      
    }

    std::unique_ptr<Pass> createMemrefStackPass() {
      return std::make_unique<MemrefStackPass>();
    }
}
