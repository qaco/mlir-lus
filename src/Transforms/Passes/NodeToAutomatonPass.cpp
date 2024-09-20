#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "../../Dialects/Lus/Node.h"
#include "../../Dialects/Lus/Instance.h"
#include "../../Dialects/Lus/InitOp.h"
#include "../../Dialects/Sync/InstOp.h"
#include "../../Dialects/Sync/OutputOp.h"
#include "../../Dialects/Sync/UndefOp.h"
#include "../../Dialects/Pssa/OutputOp.h"
#include "../../Dialects/Sync/HaltOp.h"
#include "../Utilities/Helpers.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
  namespace lus {

    struct NodeToAutomatonPass : public PassWrapper< NodeToAutomatonPass,
						OperationPass<ModuleOp>> {
      void runOnOperation() override;
    };
      
    void NodeToAutomatonPass::runOnOperation() {

      auto mod = getOperation();
      
      // lower lus::InstanceOp
      int64_t id = 2;
      mod.walk([&](lus::InstanceOp li) {
	  OpBuilder b(li);
	  StringRef n = li.getCalleeName();
	  SmallVector<Value, 4> params;
	  params.append(li.getArgOperands().begin(),
			li.getArgOperands().end());
	  SmallVector<Type, 4> resultTypes;
	  resultTypes.append(li.getResults().getTypes().begin(),
			     li.getResults().getTypes().end());
	  sync::InstOp si = b.create<sync::InstOp>(li.getLoc(),
						   n, id++,
						   params,
						   resultTypes);
	  li.replaceAllUsesWith(si);
	  li.erase();
	});

      // lower lus::NodeOp
      mod.walk([&](lus::NodeOp lNode) {
	  
	  OpBuilder b(lNode);
	  Location l = lNode.getLoc();

	  //
	  // Build sync::Node signature
	  //
	  
	  sync::NodeOp sNode=b.create<sync::NodeOp>(l,
						    lNode.getNodeName(),
						    lNode.getStaticTypes(),
						    lNode.getInputsTypes(),
						    lNode.getOutputsTypes(),
						    lNode.hasBody());

	  Block *lNodeBlock = &lNode.getBody().front();
	  Block *sNodeBlock = &sNode.getBody().front();
	  
	  //
	  // Extract structural data from lus::InitOp and remove it
	  //
	  
	  llvm::SmallSet<Operation*,4> initCode;
	  llvm::SmallMapVector<Value, Value, 4> carried;
	  lNode.walk([&](InitOp i) {
	      Helpers::feed_with_deps_of(i.getOperand(1), initCode);
	      auto p=std::pair<Value,Value>(i.getOperand(0),i.getOperand(1));
	      carried.insert(p);
	      i.getResult().replaceAllUsesExcept(i.getOperand(0),{i});
	      i.erase();
	    });
	  SmallVector<Value,4> initVals;
	  SmallVector<Type,4> initTys;
	  for (Value s: lNode.getStates()) {
	    if (!carried[s]) {
	      Type t = s.getType();
	      sync::UndefOp a = b.create<sync::UndefOp>(l, t);
	      initCode.insert(a);
	      carried[s] = a.getResult();
	    }
	    initVals.push_back(carried[s]);
	    initTys.push_back(carried[s].getType());
	  }
	  
	  //
	  // Set signature inputs
	  //
	  
	  for (Value sParam : sNode.getInputs()) {
	    Value lParam = lNodeBlock->getArgument(0);
	    lParam.replaceAllUsesWith(sParam);
	    lNodeBlock->eraseArgument(0);
	  }

	  //
	  // Produce the main loop
	  //
	  
	  OpBuilder ib = OpBuilder::atBlockEnd(sNodeBlock);

	  // The while condition : always true
	  Attribute attr = IntegerAttr::get(b.getI1Type(), true);
	  arith::ConstantOp trueOp = ib.create<arith::ConstantOp>(l, attr);
	  Value trueVal = trueOp.getResult();
	  
	  // scf::WhileOp
	  scf::WhileOp whileOp = ib.create<scf::WhileOp>(l,
							 initTys,
							 initVals);
	  ib.createBlock(&whileOp.before(), {}, initTys);

	  // scf::ConditionOp
	  ib.create<scf::ConditionOp>(l,
				      trueVal,
				      whileOp.before().getArguments());

	  // Insert the body of lus::NodeOp inside scf::WhileOp
	  Block *tmpBlock = ib.createBlock(&whileOp.after());
	  lNodeBlock->moveBefore(tmpBlock);
	  tmpBlock->erase();

	  // Replace lus::YieldOp by scf::YieldOp
	  whileOp.walk([&](lus::YieldOp y) {
	      OpBuilder yb(y);
	      Location yl = y.getLoc();
	      SmallVector<Value, 4> states;
	      states.append(y.getStates().begin(),y.getStates().end());
	      yb.create<scf::YieldOp>(yl, states);
	      y.erase();
	    });

	  //
	  // Bring back the initialization code before the main loop.
	  //
	  
	  for (Operation *op: initCode) {
	    op->moveBefore(whileOp);
	  }
	  sNodeBlock->recomputeOpOrder();

	  //
	  // Terminate the node with sync::HaltOp
	  //
	  
	  ib.setInsertionPointToEnd(sNodeBlock);
	  ib.create<sync::HaltOp>(l);

	  //
	  // Here we are !
	  //
	  
	  lNode.erase();
	});

      // lower pssa::OutputOp
      mod.walk([&](pssa::OutputOp o) {
	  OpBuilder b(o);
	  Operation *node = o.getOperation();
	  while (!isa<sync::NodeOp>(node)) {
	    node = node->getParentOp();
	  }
	  sync::NodeOp nodeOp = dyn_cast<sync::NodeOp>(node);
	  int64_t pos = o.getPosition();
	  Value sig = nodeOp.getOutput(pos);
	  Value out = o.getOperand();
	  sync::OutputOp so = b.create<sync::OutputOp>(o.getLoc(),
						       sig, out);
	  o.getResult().replaceAllUsesWith(so.getResult());
	  o.erase();
	});
    }

    std::unique_ptr<Pass> createNodeToAutomatonPass() {
      return std::make_unique<NodeToAutomatonPass>();
    }
    
  }
}
