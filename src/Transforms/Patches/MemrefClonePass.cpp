#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

    struct MemrefClonePass : public PassWrapper< MemrefClonePass,
						 OperationPass<FuncOp>> {
      void runOnOperation() override;
    };
      
    void MemrefClonePass::runOnOperation() {

      FuncOp funcOp = getOperation();

      funcOp.walk([&](memref::CloneOp cloneOp) {
	  OpBuilder b(cloneOp);
	  Value source = cloneOp.input();
	  MemRefType t = source.getType().cast<MemRefType>();
	  memref::AllocOp alloc = b.create<memref::AllocOp>(cloneOp.getLoc(),
							    t);
	  b.create<linalg::CopyOp>(cloneOp.getLoc(),
				   source,
				   alloc.getResult());
	  cloneOp.output().replaceAllUsesWith(source);
	  cloneOp.erase();
	});
      
    }

    std::unique_ptr<Pass> createMemrefClonePass() {
      return std::make_unique<MemrefClonePass>();
    }
}
