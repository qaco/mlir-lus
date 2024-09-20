#include "../../Dialects/Lus/lus.h"
#include "../../Dialects/Pssa/pssa.h"
#include "../../Dialects/Lus/WhenOp.h"
#include "../../Dialects/Lus/MergeOp.h"
#include "../../Dialects/Lus/Node.h"
#include "../../Dialects/Lus/ClockAnalysis.h"
#include "../../Dialects/Lus/ClockTree.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "../../Dialects/Sync/TickOp.h"
#include "../../Dialects/Sync/SyncOp.h"
#include "../../Dialects/Sync/UndefOp.h"
#include "mlir/Dialect/SCF/SCF.h"
#include <unordered_set>
#include <unordered_map>

using namespace mlir;
using namespace std;
using namespace mlir::arith;

namespace mlir {
  namespace lus {

    class CondactEquationsPass : public PassWrapper<
      CondactEquationsPass,
      OperationPass<NodeOp>> {
      void runOnOperation() override;
    };

    Operation* thenValueElseUndef(Location loc,
				  Value cond, Operation *op) {     
      OpBuilder b(op);
      scf::IfOp ifOp = b.create<scf::IfOp>(loc,op->getResultTypes(),
					   cond,true);
      op->replaceAllUsesWith(ifOp);
      
      // Then branch
      OpBuilder thb = ifOp.getThenBodyBuilder();
      OperationState y1State(loc, scf::YieldOp::getOperationName());
      scf::YieldOp::build(thb,y1State,op->getResults());
      Operation* y1 = thb.createOperation(y1State);
      op->moveBefore(y1);
      
      // Else branch
      OpBuilder eb = ifOp.getElseBodyBuilder();
      SmallVector<Value,4> myUndefs;
      for (Type t: op->getResultTypes()) {
	sync::UndefOp uOp = eb.create<sync::UndefOp>(loc,t);
	myUndefs.push_back(uOp.getResult());
      }
      OperationState y2State(loc, scf::YieldOp::getOperationName());
      scf::YieldOp::build(eb,y2State,myUndefs);
      eb.createOperation(y2State);
      return ifOp;
    }
    
    void CondactEquationsPass::runOnOperation() {
      ConversionTarget target(getContext());
      target.addLegalDialect<pssa::Pssa,
			     StandardOpsDialect
			     >();
      target.addIllegalDialect<Lus>();
      target.addLegalOp <
	NodeOp,
	lus::YieldOp
	>();

      NodeOp n = getOperation();

      // llvm::SmallSet<Value, 4, ValueHash> preds;
      unordered_set<Value, ValueHash> preds;

      // Save predicates
      
      n.walk([&](WhenOp w) {
	  preds.insert(w.getCondValue());
      	});

      n.walk([&](MergeOp m) {
	  preds.insert(m.getCondValue());
      	});

      ClockAnalysis analysis(n);
      assert(succeeded(analysis.analyseLogical()));
      ClockTree *tree = &analysis.getClockTree();

      //
      // Pack clocked ops
      ///

      llvm::SmallVector<Operation*,4> clockedOps;
      n.walk([&](Operation *op) {

	  if (isa<NodeOp>(op)
	      || isa<YieldOp>(op)
	      || isa<sync::SyncOp>(op)
	      || isa<sync::TickOp>(op)
	      || tree->path(op).empty())
	    return;

	  clockedOps.push_back(op);

	});

      //
      // Insert in condacts
      //

      OpBuilder nb(n.getBody());
      Type bty = IntegerType::get(nb.getContext(),1);
      Attribute zAttr = IntegerAttr::get(bty,0);
      arith::ConstantOp zOp = nb.create<arith::ConstantOp>(n.getLoc(),
							   zAttr);

      for (Operation *op: clockedOps) {
	
	Operation *risingOp = op;

	for (ClockTree::Edge edge : tree->path(op)) {
	  OpBuilder cb(op);
	  Location cl = op->getLoc();

	  Value cond_raw = edge.getData();
	  Value cond;
	  if (edge.getWhenotFlag()) {
	    // compl
	    
	    CmpIOp c = cb.create<CmpIOp>(cl,CmpIPredicate::eq,
					 cond_raw, zOp.getResult());
	    cond = c.getResult();
	  }
	  else {
	    cond = cond_raw;
	  }
	  risingOp = thenValueElseUndef(cl,cond,risingOp);
	  // If you had v = op and v was a condition, you need to update v as
	// clockTree condition(s)/edge(s) and as saved pred
	  for (auto e : llvm::zip(op->getResults(),
				  risingOp->getResults())) {
	    Value ov = get<0>(e);
	    Value nv = get<1>(e);
	    if (preds.count(ov)) {
	      tree->substitute(ov, nv);
	      preds.erase(ov);
	      preds.insert(nv);
	    }
	  }
	}
      }

      //
      // Remove lus::WhenOp
      //
      
      n.walk([&](WhenOp w) {
	  w.replaceAllUsesWith(w.getDataInput());
	  w.erase();
	});
      
      //
      // Replace lus::MergeOp by SelectOp
      //
      
      n.walk([&](MergeOp m) {
	  OpBuilder mb(m);
	  Location ml = m.getLoc();
	  
	  SelectOp s = mb.create<SelectOp>(ml, m.getCondValue(),
					   m.getTrueInput(),
					   m.getFalseInput());
	  m.getResult().replaceAllUsesWith(s.getResult());
	  m.erase();
	});
    }

    std::unique_ptr<Pass> createCondactEquationsPass() {
      return std::make_unique<CondactEquationsPass>();
    }
  }
}
