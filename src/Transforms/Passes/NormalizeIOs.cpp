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

    class NormalizeIOsPass : public PassWrapper<
      NormalizeIOsPass,
      OperationPass<NodeOp>> {
      void runOnOperation() override;
    };

    void NormalizeIOsPass::runOnOperation() {
      NodeOp nodeOp = getOperation();
      Region *body = nodeOp.getCallableRegion();
      OpBuilder b(body);
      
      SmallVector<Type,4> ninputs;
      SmallVector<Value,4> args;
      SmallVector<Type,4> atypes;
      unsigned numargs = body->getNumArguments();
      for (unsigned i = 0; i < numargs; i++) {
      	auto a = body->getArgument(i);
      	Type t = Helpers::abstract_tensor(a.getType());
	args.push_back(a);
	atypes.push_back(t);
      }
      for (auto e: llvm::zip(args,atypes)) {
	auto arg = get<0>(e);
	auto atype = get<1>(e);
	auto na = body->addArgument(atype);
	auto nna = Helpers::concretize_tensor(b,nodeOp.getLoc(),
					      arg.getType(),na);
	arg.replaceAllUsesWith(nna);
      }
      for (unsigned i = 0; i < numargs; i++) {
	body->eraseArgument(0);
      }

      YieldOp yieldOp = nodeOp.getYield();
      OpBuilder yb(yieldOp);
      SmallVector<Value,4> outs;
      SmallVector<Type,4> noutTypes;
      for (auto o: yieldOp.getOutputs()) {
	TensorType nt = Helpers::abstract_tensor(o.getType());
	outs.push_back(o);
	noutTypes.push_back(nt);
      }
      SmallVector<Value,4> nouts;
      for (auto e: llvm::zip(outs,noutTypes)) {
	auto out = get<0>(e);
	auto ntype = get<1>(e);
	Value nout = Helpers::abstractize_tensor(yb, yieldOp.getLoc(),
						 ntype,out);
	nouts.push_back(nout);
      }
      SmallVector<Value,4> states;
      for (auto s: yieldOp.getStates())
	states.push_back(s);
      yb.create<YieldOp>(yieldOp.getLoc(),states,nouts);
      yieldOp.erase();

      auto type = NodeType::get(nodeOp.getContext(),
    				nodeOp.getStaticTypes(),
    				atypes,
    				nodeOp.getStatesTypes(),
    				noutTypes,
    				nodeOp.getType().getRegionKind());
      nodeOp.getOperation()->setAttr(nodeOp.getTypeAttrName(),
				     TypeAttr::get(type));
    }
    
    std::unique_ptr<Pass> createNormalizeIOsPass() {
      return std::make_unique<NormalizeIOsPass>();
    }
  }
}
