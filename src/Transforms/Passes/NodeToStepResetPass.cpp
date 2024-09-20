#include <list>

#include "../../Dialects/Lus/lus.h"
#include "../../Dialects/Lus/NodeTest.h"
#include "../../Dialects/Sync/UndefOp.h"
#include "../../Dialects/Lus/InitOp.h"
#include "../Utilities/Helpers.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
  namespace lus {

    class NodeToStepResetPass : public PassWrapper < NodeToStepResetPass,
						      OperationPass<ModuleOp> > {
      void runOnOperation() override;
    };

    void NodeToStepResetPass::runOnOperation() {
      ConversionTarget target(getContext());
      target.addIllegalDialect<Lus>();

      
      ModuleOp m(getOperation());
      m.walk([&](NodeTestOp n) {

	  OpBuilder topBuilder(n);
	  Location topLoc = n.getLoc();
	  StringRef nname = n.sym_name();
	  auto context = topBuilder.getContext();
	  StringAttr visibility = StringAttr::get(context,"public");
	  StringAttr stepName = StringAttr::get(context,nname + "_step");
	  StringAttr resName = StringAttr::get(context,nname + "_reset");

	  //
	  // Generate the step function
	  //
	       
	  NodeTestOp stepBase = n.clone();

	  // Remove initializations
	  stepBase.walk([&](InitOp i) {
	      i.getResult().replaceAllUsesExcept(i.getOperand(0),{i});
	      i.erase();
	    });

	  // Replace yields by returns
	  stepBase.walk([&](YieldOp y) {
	      OpBuilder b(y);
	      Location loc = y.getLoc();
	      SmallVector<Value, 4> retOps;
	      retOps.append(y.getOutputs().begin(), y.getOutputs().end());
	      retOps.append(y.getStates().begin(), y.getStates().end());
	      b.create<ReturnOp>(loc, retOps);
	      y.erase();
	    });	       

	  // Build the step function and take the body of the initial
	  // node
	  SmallVector<Type, 4> stepInTys;
	  stepInTys.append(stepBase.getInputTypes().begin(),
			   stepBase.getInputTypes().end());
	  stepInTys.append(stepBase.getStateTypes().begin(),
			   stepBase.getStateTypes().end());
	  SmallVector<Type, 4> stepOutTys;
	  stepOutTys.append(stepBase.getOutputTypes().begin(),
			    stepBase.getOutputTypes().end());
	  stepOutTys.append(stepBase.getStateTypes().begin(),
			    stepBase.getStateTypes().end());
	  FunctionType stepTy = topBuilder.getFunctionType(stepInTys,
							   stepOutTys);
	       
	  FuncOp step = topBuilder.create<FuncOp>(topLoc,
						  stepName,
						  TypeAttr::get(stepTy),
						  visibility);
	  Region &b = stepBase.getBody();
	  while (b.getNumArguments() >
		 stepBase.getNumInputs() + stepBase.getCardState()) {
	    unsigned i = stepBase.getNumInputs();
	    b.eraseArgument(i);
	  }
	  step.getBody().takeBody(stepBase.getBody());
	  
	  // Here we are
	  stepBase.erase();

	  //
	  // Generate the reset function
	  //

	  NodeTestOp resetBase = n.clone();
	  llvm::SmallSet<Operation*,4> usefulOps;
	  llvm::SmallMapVector<Value, Value, 4> returnVals;
	  
	  // Remember initializations
	  resetBase.walk([&](InitOp i) {
	      Helpers::feed_with_deps_of(i.getOperand(1), usefulOps);
	      auto p=std::pair<Value,Value>(i.getOperand(0),i.getOperand(1));
	      returnVals.insert(p);
	    });

	  // Remember values to return
	  SmallVector<Value,4> rets;
	  YieldOp resetY = resetBase.getYield();
	  for (Value s: resetBase.getState()) {
	    if (!returnVals[s]) {
	      OpBuilder yb(resetY);
	      Location yl = resetY.getLoc();
	      Type t = s.getType();
	      sync::UndefOp a = yb.create<sync::UndefOp>(yl, t);
	      usefulOps.insert(a);
	      returnVals[s] = a.getResult();
	    }
	    rets.push_back(returnVals[s]);
	  }

	  SmallVector<Type, 4> resOutTys;
	  resOutTys.append(resetBase.getStateTypes().begin(),
			   resetBase.getStateTypes().end());
	  FunctionType resTy = topBuilder.getFunctionType({},resOutTys);

	  // Build the reset function and take the body of the initial
	  // node
	  FuncOp reset = topBuilder.create<FuncOp>(topLoc,
						   resName,
						   TypeAttr::get(resTy),
						   visibility);
	  reset.getBody().takeBody(resetBase.getBody());

	  // Add return and remove all the useless code
	  Operation &fOp = reset.getBody().front().front();
	  reset.getBody().front().splitBlock(&fOp);
	       
	  OpBuilder resB(reset.getBody());
	  ReturnOp returnOp = resB.create<ReturnOp>(topLoc, rets);
	       
	  for (Operation *op: usefulOps) {
	    op->moveBefore(returnOp);
	  }
	  reset.getBody().back().erase();

	  // Normalize
	  while (reset.getBody().getNumArguments() > 0) {
	    reset.getBody().eraseArgument(0);
	  }
	  reset.getBody().front().recomputeOpOrder();
	       
	  // Here we are
	  resetBase.erase();

	  //
	  // Remove the initial node
	  //
	       
	  n.erase();

	});
    }

    std::unique_ptr<Pass> createNodeToStepResetPass() {
      return std::make_unique<NodeToStepResetPass>();
    }
  }
}
