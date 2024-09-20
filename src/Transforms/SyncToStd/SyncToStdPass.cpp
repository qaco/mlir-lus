#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
#include "../../Dialects/Sync/SyncOp.h"
#include "../../Dialects/Sync/Node.h"
#include "../../Dialects/Sync/InputOp.h"
#include "../../Dialects/Sync/TickOp.h"
#include "../../Dialects/Sync/SelectOp.h"
#include "../../Dialects/Sync/UndefOp.h"
#include "../../Dialects/Sync/HaltOp.h"
#include "NodeToFun.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

namespace mlir {
  namespace sync {

    struct SyncToStandardPass : public PassWrapper< SyncToStandardPass,
						    OperationPass<ModuleOp>> {
      void runOnOperation() override;
    };
      
    void SyncToStandardPass::runOnOperation() {

      auto mod = getOperation();

      NodeToFun nodeToFun;
      nodeToFun(mod);

      // lower sync::SyncOp
      mod.walk([&](sync::SyncOp s) {
		 OpBuilder b(s);
		 OperandRange values = s.getValues();
		 for (auto e: llvm::zip(s.getResults(), values)) {
		   get<0>(e).replaceAllUsesWith(get<1>(e));
		 }
		 s.erase();
	       });
      
      // lower sync::TickOp
      OpBuilder mb(&mod.getBodyRegion());
      TypeAttr ft = TypeAttr::get(mb.getFunctionType({},
						     mb.getI32Type()));
      StringAttr v = mb.getStringAttr("private") ;
      StringAttr n = mb.getStringAttr(sync::TickOp::getFunctionName()) ;
      FuncOp f = mb.create<FuncOp>(mod.getLoc(),n,ft,v);
      mod.walk([&](sync::TickOp t) {
		 OpBuilder b(t);
		 b.create<CallOp>(t.getLoc(),f);
		 t.erase();
	       });
      
      
      // lower sync::HaltOp
      mod.walk([&](sync::HaltOp h) {
		 OpBuilder b(h);
		 b.create<ReturnOp>(h.getLoc());
		 h.erase();
	       });
      
      // lower sync::UndefOp
      mod.walk([&](sync::UndefOp u) {
		 Location loc = u.getLoc();
		 OpBuilder b(u);
		 Type t = u.getResult().getType();
		 Value r;
		 if (auto tt = t.dyn_cast<TensorType>()) {
		   ArrayRef<int64_t> shape = tt.getShape();
		   Type eltType = tt.getElementType();
		   Attribute tensorAttr;
		   if (tt.isa<IntegerType>()) {
		     Attribute eltAttr = IntegerAttr::get(bty, 0);
		     tensorAttr = DenseIntElementsAttr::get(tt,
							      eltAttr);
		   }
		   else if (tt.isa<Float32Type>()) {
		     Attribute eltAttr = FloatAttr::get(bty, 0);
		     tensorAttr = DenseFloatElementsAttr::get(tt,
							      eltAttr);
		   }
		   arith::ConstantOp tensorOp = b.create<arith::ConstantOp>(loc,
									    tensorAttr);
		   // linalg::InitTensorOp initTensorOp = b.create
		   //   <linalg::InitTensorOp>(loc,shape,eltType);
		   r = tensorOp.getResult();
		 }
		 else if (!t.isa<ShapedType>()) {
		   // Create t: tensor<1xt>
		   int64_t shape[1] = {1};
		   linalg::InitTensorOp initTensorOp = b.create
		     <linalg::InitTensorOp>(loc,shape,t);
		   Value t = initTensorOp.getResult();

		   // Create i = 0: index
		   Attribute az = IntegerAttr::get(b.getIndexType(), 0);
		   arith::ConstantOp oz = b.create
		     <arith::ConstantOp>(loc, az);
		   Value z = oz.getResult();

		   // Extract t[0]
		   tensor::ExtractOp extractOp = b.create
		     <tensor::ExtractOp>(loc, t, z);
		   r = extractOp.getResult();
		 }
		 u.getResult().replaceAllUsesWith(r);
		 u.erase();
	       });
      
      mod.walk([&](sync::SelectOp s) {
		 OpBuilder b(s);
		 mlir::SelectOp ns =  b.create
		   <mlir::SelectOp>(s.getOperation()->getLoc(),
			      s.getCondition(),
			      s.getTrueBranch(),
			      s.getFalseBranch());
		 s.getOperation()->replaceAllUsesWith(ns);
		 s.erase();
	       });
    }

    std::unique_ptr<Pass> createSyncToStandardPass() {
      return std::make_unique<SyncToStandardPass>();
    }
    
  }
}
