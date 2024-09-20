#include "../../Dialects/Lus/lus.h"
#include "../../Dialects/Pssa/pssa.h"
#include "../../Dialects/Lus/WhenOp.h"
#include "../../Dialects/Sync/UndefOp.h"
#include "../../Dialects/Lus/MergeOp.h"
#include "../../Dialects/Lus/PreOp.h"
#include "../../Dialects/Lus/FbyOp.h"
#include "../../Dialects/Lus/Node.h"
#include "../../Dialects/Lus/InitOp.h"
#include "../../Dialects/Lus/KPeriodicOp.h"
#include "../../Dialects/Lus/ClockAnalysis.h"
#include "../Utilities/Helpers.h"
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

    class NormalizeSmarterPass : public PassWrapper<
      NormalizeSmarterPass,
      OperationPass<NodeOp>> {
      void runOnOperation() override;
    private:
      Operation *node;
      Operation *baseTimer;
      FbyOp getBaseTimer(NodeOp n);
      FbyOp createTimer(NodeOp n);
      FbyOp getTimer(NodeOp n, ClockTree::Path p);
      void lowerPre(NodeOp n);
      void lowerInputDependentFby(NodeOp n);
      void lowerSingletonKp(NodeOp n);
      void lowerPairKp(NodeOp n);
      void lowerHeadlessKp(NodeOp n);
      void lowerGeneralKp(NodeOp n);
      void normalizeClockedFby(NodeOp n);
      void lowerFby(NodeOp n);
    };

    Value preToExplicitState(NodeOp n, Value os) {
      
      // Instead of pre result, an explicit state of the node
      Value s = n.addState(os.getType());

      // yield the explicit state at cycle k > 0
      YieldOp oy = n.getYield();
      SmallVector<Value, 4> states;
      states.append(oy.getStates().begin(),
		    oy.getStates().end());
      states.push_back(os);
      SmallVector<Value, 4> outputs;
      outputs.append(oy.getOutputs().begin(),
		     oy.getOutputs().end());
      oy.addState(os);
      
      return s;
    }
    
    FbyOp NormalizeSmarterPass::getBaseTimer(NodeOp n) {
      if (n.getOperation() == this->node) {
	return dyn_cast<FbyOp>(this->baseTimer);
      }
      else {
	FbyOp f = createTimer(n);
	this->node = n.getOperation();
	this->baseTimer = f.getOperation();
	return f;
      }
    }
    
    FbyOp NormalizeSmarterPass::createTimer(NodeOp n) {
      OpBuilder nb(n.getBody());
      Location rLoc = n.getBody().getLoc();
      Type ity = IndexType::get(nb.getContext());
      Attribute zeroAttr = IntegerAttr::get(ity,0);
      Attribute oneAttr = IntegerAttr::get(ity,1);
      arith::ConstantOp zero = nb.create<arith::ConstantOp>(rLoc,
							    zeroAttr);
      arith::ConstantOp one = nb.create<arith::ConstantOp>(rLoc,
							   oneAttr);
      FbyOp fbyOp = nb.create<FbyOp>(rLoc, zero, zero);
      AddIOp addiOp = nb.create<AddIOp>(rLoc,
					fbyOp.getResult(),
					one.getResult());
      fbyOp.getOperation()->setOperand(1,addiOp.getResult());
      return fbyOp;
    }

    FbyOp NormalizeSmarterPass::getTimer(NodeOp n, ClockTree::Path p) {
      if (p.empty()) {
	return getBaseTimer(n);
      }
      else {
	return createTimer(n);
      }
    }

    void NormalizeSmarterPass::lowerPre(NodeOp n) {
      n.walk([&](PreOp p) {
	  OpBuilder b(p);
	  Location loc = p.getLoc();

	  sync::UndefOp u = b.create<sync::UndefOp>(loc,
						      p.getResult().getType());
	  FbyOp fbyOp = b.create<FbyOp>(loc, u.getResult(),
					p.getOperand());
	  KperiodicOp kpOp = b.create<KperiodicOp>(loc, "0(1)");
	  Cond<Value> clk(kpOp.getResult(),false);
	      
	  WhenOp whenOp = b.create<WhenOp>(loc, clk, fbyOp.getResult());
	  p.getResult().replaceAllUsesWith(whenOp.getResult());
	  p.erase();
	});
      return;
    }

    void NormalizeSmarterPass::lowerInputDependentFby(NodeOp n) {
      n.walk([&](FbyOp fbyOp) {
	  if (!Helpers::depends_on_inputs(fbyOp.getLhs()))
	    return;
	  OpBuilder b(fbyOp);
	  Location loc = fbyOp.getLoc();
	    
	  KperiodicOp kpOp = b.create<KperiodicOp>(loc, "0(1)");
	  // True if time = 0
	  Cond<Value> t0(kpOp.getResult(),true);
	  // True if time > 0
	  Cond<Value> tn(kpOp.getResult(),false);

	  Value s = preToExplicitState(n, fbyOp.getRhs());
	  WhenOp initWhenOp = b.create<WhenOp>(loc,t0,fbyOp.getLhs());
	  WhenOp nextWhenOp = b.create<WhenOp>(loc,tn,s);
	  MergeOp mergeOp = b.create<MergeOp>(loc,tn,
					      nextWhenOp.getResult(),
					      initWhenOp.getResult());

	  fbyOp.getResult().replaceAllUsesWith(mergeOp.getResult());
	  fbyOp.erase();
	});
      return;
    }

    void NormalizeSmarterPass::lowerSingletonKp(NodeOp n) {
      n.walk([&](KperiodicOp kp) {

	  if (!kp.isSingleton())
	    return;

	  OpBuilder b(kp);
	  Location loc = kp.getLoc();
	  Type bty = IntegerType::get(b.getContext(),1);

	  Attribute attr = IntegerAttr::get(bty, kp.getPeriod().front());
	  arith::ConstantOp kOp = b.create<arith::ConstantOp>(loc,attr);

	  kp.getResult().replaceAllUsesWith(kOp.getResult());
	  kp.erase();
	});
      return;
    }

    void NormalizeSmarterPass::lowerPairKp(NodeOp n) {
	n.walk([&](KperiodicOp kp) {

	  if (!kp.isPair())
	    return;

	  bool left = kp.getPrefix().front();
	  bool right = kp.getPeriod().front();

	  OpBuilder b(kp);
	  Location loc = kp.getLoc();
	  Type bty = IntegerType::get(b.getContext(),1);

	  Attribute lAttr = IntegerAttr::get(bty, left);
	  Attribute rAttr = IntegerAttr::get(bty, right);
	  arith::ConstantOp lOp = b.create<arith::ConstantOp>(loc,
							       lAttr);
	  arith::ConstantOp rOp = b.create<arith::ConstantOp>(loc,
							      rAttr);
	  FbyOp fbyOp = b.create<FbyOp>(loc,
					lOp.getResult(),
					rOp.getResult());
	  kp.getResult().replaceAllUsesWith(fbyOp.getResult());
	  kp.erase();
	});
	return;
    }

    void NormalizeSmarterPass::lowerHeadlessKp(NodeOp n) {
      ClockAnalysis analysis(n);
      analysis.analyse();
      ClockTree *clockTree = &analysis.getClockTree();
      n.walk([&](KperiodicOp kp) {

	  if (!kp.isHeadLess())
	    return;

	  OpBuilder b(kp);
	  Location loc = kp.getLoc();
	  Type bty = IntegerType::get(b.getContext(),1);
	  Type ity = IndexType::get(b.getContext());
	  FbyOp fbyOp = getTimer(n, clockTree->path(kp.getOperation()));

	  // Generate the period tensor
	  std::vector<bool> period = kp.getPeriod();
	  std::vector<Attribute> periodAsAttrs;
	  for (bool v : period) {
	    Attribute attr = IntegerAttr::get(bty, v);
	    periodAsAttrs.push_back(attr);
	  }
	  std::vector<int64_t> periodShape;
	  periodShape.push_back(period.size());
	  ShapedType periodType = RankedTensorType::get(periodShape,bty);
	  Attribute periodAttr = DenseIntElementsAttr::get(periodType,
							   periodAsAttrs);
	  arith::ConstantOp peOp = b.create<arith::ConstantOp>(loc,
							       periodAttr);
	  Attribute peSizeAttr = IntegerAttr::get(ity,period.size());
	  arith::ConstantOp peSize = b.create<arith::ConstantOp>(loc,
								 peSizeAttr);
	  RemUIOp rOp = b.create<RemUIOp>(loc,
					  fbyOp.getResult(),
					  peSize.getResult());
	  ExtractOp r2=b.create<ExtractOp>(loc,
					   peOp.getResult(),
					   rOp.getResult());

	  // Finish
	  kp.getResult().replaceAllUsesWith(r2.getResult());
	  kp.erase();
	});
      return;
    }

    void NormalizeSmarterPass::lowerGeneralKp(NodeOp n) {
      ClockAnalysis analysis(n);
      analysis.analyse();
      ClockTree *clockTree = &analysis.getClockTree();
      
      n.walk([&](KperiodicOp kp) {

	  OpBuilder b(kp);
	  Location loc = kp.getLoc();
	  Type bty = IntegerType::get(b.getContext(),1);
	  Type ity = IndexType::get(b.getContext());
	  FbyOp fbyOp = getTimer(n, clockTree->path(kp.getOperation()));

	  // Generate the prefix tensor
	  std::vector<bool> prefix = kp.getPrefix();
	  std::vector<Attribute> prefixAsAttrs;
	  for (bool v : prefix) {
	    Attribute attr = IntegerAttr::get(bty, v);
	    prefixAsAttrs.push_back(attr);
	  }
	  std::vector<int64_t> prefixShape;
	  prefixShape.push_back(prefix.size());
	  ShapedType prefixType = RankedTensorType::get(prefixShape,
							bty);
	  Attribute prefixAttr = DenseIntElementsAttr::get(prefixType,
							   prefixAsAttrs);
	  arith::ConstantOp prOp = b.create<arith::ConstantOp>(loc,
							       prefixAttr);

	  // Generate the period tensor
	  std::vector<bool> period = kp.getPeriod();
	  std::vector<Attribute> periodAsAttrs;
	  for (bool v : period) {
	    Attribute attr = IntegerAttr::get(bty, v);
	    periodAsAttrs.push_back(attr);
	  }
	  std::vector<int64_t> periodShape;
	  periodShape.push_back(period.size());
	  ShapedType periodType = RankedTensorType::get(periodShape,bty);
	  Attribute periodAttr = DenseIntElementsAttr::get(periodType,
							   periodAsAttrs);
	  arith::ConstantOp peOp = b.create<arith::ConstantOp>(loc,
							       periodAttr);

	  // Generate a condition
	  Attribute prSizeAttr = IntegerAttr::get(ity,prefix.size());
	  arith::ConstantOp prSize = b.create<arith::ConstantOp>(loc,
								 prSizeAttr);
	  CmpIOp c = b.create<CmpIOp>(loc,
				      CmpIPredicate::ult,
				      fbyOp.getResult(),
				      prSize.getResult());

	  // Generate ifop
	  scf::IfOp ifOp = b.create<scf::IfOp>(loc,bty,c,true);
	  // then
	  OpBuilder thb = ifOp.getThenBodyBuilder();
	  ExtractOp r1=thb.create<ExtractOp>(loc,
					     prOp.getResult(),
					     fbyOp.getResult());
	  OperationState y1State(loc, scf::YieldOp::getOperationName());
	  scf::YieldOp::build(thb,y1State,{r1});
	  thb.createOperation(y1State);
	  //else
	  OpBuilder eb = ifOp.getElseBodyBuilder();
	  SubIOp subiOp = eb.create<SubIOp>(loc,
					    fbyOp.getResult(),
					    prSize.getResult());
	  Value subi = subiOp.getResult();
	  Attribute peSizeAttr = IntegerAttr::get(ity,period.size());
	  arith::ConstantOp peSize = eb.create<arith::ConstantOp>(loc,
								  peSizeAttr);
	  RemUIOp rOp =  eb.create<RemUIOp>(loc,
					    subi,
					    peSize.getResult());
	  ExtractOp r2=eb.create<ExtractOp>(loc,
					    peOp.getResult(),
					    rOp.getResult());
	  OperationState y2State(loc, scf::YieldOp::getOperationName());
	  scf::YieldOp::build(eb,y2State,{r2});
	  eb.createOperation(y2State);

	  // Finish
	  kp.getResult().replaceAllUsesWith(ifOp.getResult(0));

	  kp.erase();
	});
      return;
    }

    void NormalizeSmarterPass::normalizeClockedFby(NodeOp n) {
      bool fixedPoint;
      do {
	ClockAnalysis analysis(n);
	analysis.analyse();
	ClockTree *clockTree = &analysis.getClockTree();
	fixedPoint = true;
	n.walk([&](FbyOp fbyOp) {
	    ClockTree::Path p = clockTree->path(fbyOp.getOperation());
	    if (p.empty())
	      return;
	    Location loc = fbyOp.getLoc();
	    OpBuilder builder = OpBuilder(fbyOp);
	    Cond<Value> cond = p.back();
	    Cond<Value> condNot(cond.getData(),!cond.getWhenotFlag());
	    
	    WhenOp when1Op = builder.create<WhenOp>(loc,cond,
						    fbyOp.getResult());
	    fbyOp.getResult().replaceAllUsesExcept(when1Op.getResult(),
						   {when1Op});
	    WhenOp when2Op = builder.create<WhenOp>(loc,condNot,
						    fbyOp.getResult());

	    Value rhsFby = fbyOp.getRhs();
	    MergeOp mergeOp = builder.create<MergeOp>(loc,cond,
						      rhsFby,
						      when2Op.getResult());
	    fbyOp.getOperation()->setOperand(1,mergeOp.getResult());

	    fixedPoint = false;
	  });
      }	while (!fixedPoint);
      return;
    }

    void NormalizeSmarterPass::lowerFby(NodeOp n) {
      n.walk([&](FbyOp f) {
	  OpBuilder b(f);
	  Location loc = f.getLoc();

	  // next value
	  Value s = preToExplicitState(n, f.getRhs());
	  f.getResult().replaceAllUsesWith(s);

	  // initial value
	  InitOp initOp = b.create<InitOp>(loc,s,f.getLhs());
	  s.replaceAllUsesExcept(initOp.getResult(),{initOp});
	  f.erase();
	});
      return;
    }


    Value boolConstant(OpBuilder &b, Location loc, bool v) {
      Attribute attr = IntegerAttr::get(b.getI1Type(), v);
      arith::ConstantOp op = b.create<arith::ConstantOp>(loc, attr);
      return op.getResult();
    }
    
    void NormalizeSmarterPass::runOnOperation() {
      ConversionTarget target(getContext());
      target.addLegalDialect<pssa::Pssa,
			     StandardOpsDialect
			     >();
      target.addIllegalDialect<Lus>();
      target.addLegalOp <
	WhenOp,
	MergeOp,
	NodeOp,
	PreOp,
	lus::YieldOp
	>();

      NodeOp n = getOperation();

      if (!n.hasBody())
      	return;

      // pre -> fby + undef + merge kp
      lowerPre(n);
      // fby -> fby + merge kp
      normalizeClockedFby(n);
      // fby -> explicit state + merge kp
      lowerInputDependentFby(n);
      // "(X)" -> const
      lowerSingletonKp(n);
      // X(Y)" -> fby
      lowerPairKp(n);
      // (Y+)" -> fby on time indices + tensor
      lowerHeadlessKp(n);
      // X+(Y+)" -> fby on time indices + tensors
      lowerGeneralKp(n);
      // Again
      normalizeClockedFby(n);
      // fby -> explicit state + init
      lowerFby(n);
      // TODO : should not rely on our code
      SortAlongClocks sortAlongClocks;
      sortAlongClocks(n);
      // std::error_code err;
      // llvm::raw_fd_ostream stream("/dev/stdout", err);
      // stream << "\nDEBUG\n" << n << "\nDEBUG\n";
      // n.getBody().front().recomputeOpOrder();
      // n.forceDominance();
    }

    std::unique_ptr<Pass> createNormalizeSmarterPass() {
      return std::make_unique<NormalizeSmarterPass>();
    }
  }
}
