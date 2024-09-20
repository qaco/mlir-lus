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

    struct MemrefCopyPass : public PassWrapper< MemrefCopyPass,
						 OperationPass<FuncOp>> {
      void runOnOperation() override;
    };
      
    void MemrefCopyPass::runOnOperation() {

      FuncOp funcOp = getOperation();

      funcOp.walk([&](memref::CopyOp copyOp) {
	  OpBuilder b(copyOp);
	  Value source = copyOp.source();
	  Value target = copyOp.target();
	  b.create<linalg::CopyOp>(copyOp.getLoc(),
				   source,
				   target);
	  copyOp.erase();
	});
      
    }

    std::unique_ptr<Pass> createMemrefCopyPass() {
      return std::make_unique<MemrefCopyPass>();
    }
}
