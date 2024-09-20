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
#include "../SyncToStd/NodeToFun.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

namespace mlir {
  namespace sync {

    struct SyncToStandardPass : public PassWrapper< SyncToStandardPass,
						    OperationPass<ModuleOp>> {
      void runOnOperation() override;
    };

    Attribute undefConstOf(Type eltType) {
      Attribute constAtt;
      if (eltType.isF32()) {
	constAtt = FloatAttr::get(eltType, 0);
      }
      else if (eltType.isInteger(32)
	       || eltType.isInteger(64)
	       || eltType.isInteger(1)) {
	constAtt = IntegerAttr::get(eltType, 0);
      }
      else {
	assert(false);
      }
      return constAtt;
    }

    unsigned argument_position(Value arg, FuncOp f,
			       unsigned offset) {
      unsigned i = 0;
      for (Value a : f.getArguments().drop_front(offset)) {
	if (a == arg)
	  return i;
	i++;
      }
      assert(false);
    }
    
    void SyncToStandardPass::runOnOperation() {
      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stdout", err);


      auto mod = getOperation();
      NodeToFun nodeToFun;
      nodeToFun(mod);

      OpBuilder utlb(mod.getBodyRegion());
      const Type instTy = utlb.getI32Type();
      const Type posTy = utlb.getI32Type();
      const Type dimTy = utlb.getI32Type();
      const Type sizeTy = utlb.getI32Type();
      
      const Type syncTy = utlb.getI32Type();
      
      StringAttr visibility = utlb.getStringAttr("private") ;
      
      // lower sync::instOp

      mod.walk([&](sync::InstOp instOp) {

	  OpBuilder tlb(mod.getBodyRegion());
	  StringRef nname = instOp.getCalleeName();
	  
	  // Signals functions
	  
	  SmallVector<Type, 4> inputTTypes;
	  for (auto v: instOp.getArgOperands()) {
	    Type tt = Helpers::abstract_tensor(v.getType());
	    inputTTypes.push_back(tt);
	  }
	  SmallVector<Type, 4> outputTTypes;
	  for (auto v: instOp.getResults()) {
	    Type tt = Helpers::abstract_tensor(v.getType());
	    outputTTypes.push_back(tt);
	  }

	  SmallVector<FuncOp, 4> signals;
	  std::string read_prefix = "sched_read_input_";
	  for (Type t: inputTTypes) {
	    std::string name = read_prefix + Helpers::printable(t);
	    StringAttr fname = StringAttr::get(tlb.getContext(),
					       name);
	    TypeAttr ft = TypeAttr::get(tlb.getFunctionType({instTy,t}, {}));
	    FuncOp cf = tlb.create<FuncOp>(instOp.getLoc(),
					   fname, ft, visibility);
	    signals.push_back(cf);
	  }

	  std::string write_prefix = "sched_write_output_";
	  for (Type t: outputTTypes) {
	    std::string name = write_prefix + Helpers::printable(t);
	    StringAttr fname = StringAttr::get(tlb.getContext(),
					       name);
	    TypeAttr ft = TypeAttr::get(tlb.getFunctionType({instTy,t},
							    {syncTy}));
	    FuncOp cf = tlb.create<FuncOp>(instOp.getLoc(),
					   fname, ft, visibility);
	    signals.push_back(cf);
	  }

	  // Start function

	  TypeAttr startt = TypeAttr::get(tlb.getFunctionType({instTy},{}));
	  llvm::Twine startname = nname+"_start";
	  StringAttr astartname = StringAttr::get(tlb.getContext(),
						 startname);
	  FuncOp startf = tlb.create<FuncOp>(instOp.getLoc(),
					     astartname, startt,
					     visibility);
	  Block* startBlock = startf.addEntryBlock();
	  SmallVector<Value, 4> coreParams;
	  coreParams.push_back(startBlock->getArgument(0));
	  OpBuilder startB(startf.getBody());
	  for (auto sig: signals) {
	    Attribute a = FlatSymbolRefAttr::get(startB.getContext(),
						 sig.sym_name());
	    ConstantOp op = startB.create<ConstantOp>(instOp.getLoc(),
						      sig.getType(),
						      a);
	    coreParams.push_back(op.getResult());
	  }
	  OperationState callCoreState(instOp.getLoc(),
				       CallOp::getOperationName());
	  CallOp::build(startB, callCoreState, nname, {}, coreParams);
	  startB.createOperation(callCoreState);
	  startB.create<ReturnOp>(instOp.getLoc());

	  // set communication buffers (tensors ?)
	  
	  std::string set_input_prefix = "sched_set_input_";
	  for (Type t: inputTTypes) {
	    std::string name = set_input_prefix + Helpers::printable(t);
	    StringAttr fname = StringAttr::get(tlb.getContext(),
					       name);
	    FunctionType ft = tlb.getFunctionType({instTy,posTy,
						   dimTy,sizeTy,t}, {});
	    TypeAttr aft = TypeAttr::get(ft);
	    FuncOp cf = tlb.create<FuncOp>(instOp.getLoc(),
					   fname, aft, visibility);
	  }
	  
	  std::string set_output_prefix = "sched_set_output_";
	  for (Type t: outputTTypes) {
	    std::string name = set_output_prefix + Helpers::printable(t);
	    StringAttr fname = StringAttr::get(tlb.getContext(),
					       name);
	    FunctionType ft = tlb.getFunctionType({instTy,posTy,
						   dimTy,sizeTy,t}, {});
	    TypeAttr aft = TypeAttr::get(ft);
	    FuncOp cf = tlb.create<FuncOp>(instOp.getLoc(),
					   fname, aft, visibility);
	  }
	  
	  // Inst function

	  SmallVector<Type,4> paramsInst;
	  paramsInst.push_back(instTy);
	  paramsInst.append(inputTTypes);
	  paramsInst.append(outputTTypes);
	  TypeAttr instt = TypeAttr::get(tlb.getFunctionType({paramsInst},
							     {}));
	  llvm::Twine instname = nname+"_inst";
	  StringAttr ainstname = StringAttr::get(tlb.getContext(),
						 instname);
	  FuncOp instf = tlb.create<FuncOp>(instOp.getLoc(),
					    ainstname, instt,
					    visibility);
	  Block* instBlock = instf.addEntryBlock();
	  OpBuilder instB(instf.getBody());
	  Attribute startPtrName = FlatSymbolRefAttr::get(startB.getContext(),
							  startf.sym_name());
	  ConstantOp op = instB.create<ConstantOp>(instOp.getLoc(),
						   startf.getType(),
						   startPtrName);
	});

      // stream << mod;
	  
      // lower sync::nodeOp
      
      mod.walk([&](sync::NodeOp nodeOp) {

	  OpBuilder tlb(mod.getBodyRegion());

	  const unsigned off_out=nodeOp.getNumInputs()+1;
	  StringRef nname = nodeOp.getNodeName();
	  
	  // Translate and pack the signal types
	  SmallVector<Type, 4> transIoTys;
	  transIoTys.push_back(instTy); // The inst id
	  for (Value i : nodeOp.getInputs()) {
	    SiginType st = i.getType().cast<SiginType>();
	    Type t = st.getType();
	    FunctionType ft = tlb.getFunctionType({posTy}, {t});
	    transIoTys.push_back(ft);
	  }
	  for (Value i : nodeOp.getOutputs()) {
	    SigoutType st = i.getType().cast<SigoutType>();
	    Type t = st.getType();
	    FunctionType ft = tlb.getFunctionType({posTy, t}, {syncTy});
	    transIoTys.push_back(ft);
	  }

	  // Build the core function
	  TypeAttr cft = TypeAttr::get(tlb.getFunctionType(transIoTys, {}));
	  StringAttr cfname = StringAttr::get(tlb.getContext(),
					      nname);
	  FuncOp cf = tlb.create<FuncOp>(nodeOp.getLoc(),
	  				 cfname, cft, visibility);

	  // Region stuff
	  nodeOp.getBody().insertArgument((unsigned)0, instTy);
	  cf.getBody().takeBody(nodeOp.getBody());
	  for (unsigned i = 1 ; i < transIoTys.size() ; i++) {
	    Type transt = transIoTys[i];
	    Value neoArg = cf.getBody().insertArgument(i, transt);
	    Value oldArg = cf.getBody().getArgument(i + 1);
	    oldArg.replaceAllUsesWith(neoArg);
	    cf.getBody().eraseArgument(i + 1);
	  }

	  
	  // Lower sync::InputOp
	  cf.walk([&](InputOp inp) {
	      OpBuilder inpb(inp);
	      Location inpl = inp.getLoc();
	      Value myFun = inp.getSignal();
	      unsigned i = argument_position(myFun, cf, 1);
	      IntegerAttr posAttr = IntegerAttr::get(posTy, i);
	      arith::ConstantOp posop=inpb.create<arith::ConstantOp>(inpl,
								     posAttr);
	      Value pos = posop.getResult();
	      OperationState callState(inpl,
				       CallIndirectOp::getOperationName());
	      CallIndirectOp::build(inpb, callState, myFun, {pos});
	      Operation *call = inpb.createOperation(callState);
	      inp.replaceAllUsesWith(call);
	      inp.erase();
	    });

	  
	  // Lower sync::OutputOp
	cf.walk([&](OutputOp outp) {
	    OpBuilder outpb(outp);
	    Location outpl = outp.getLoc();
	    Value myFun = outp.getSignal();
	    unsigned i = argument_position(myFun, cf, off_out);
	    IntegerAttr posAttr = IntegerAttr::get(posTy, i);
	    arith::ConstantOp posop=outpb.create<arith::ConstantOp>(outpl,
								    posAttr);
	    Value pos = posop.getResult();
	    OperationState callState(outpl,
				     CallIndirectOp::getOperationName());
	    CallIndirectOp::build(outpb, callState, myFun,
				  {pos, outp.getParameter()});
	    Operation *call = outpb.createOperation(callState);
	    outp.replaceAllUsesWith(call);
	    outp.erase();
	  });
	nodeOp.erase();
	});
	  

	  
      
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

	  if (!t.isa<ShapedType>()) {
	    Attribute constAtt = undefConstOf(t);
	    arith::ConstantOp op = b.create<arith::ConstantOp>(loc,constAtt);
	    u.getResult().replaceAllUsesWith(op.getResult());
	    u.erase();
	    return;
	  }

	  // Shape/element type of the future tensor
	  bool dyn_shape = false;
	  std::vector<int64_t> shape;
	  Type eltType;
	  if (auto tt = t.dyn_cast<TensorType>()) {
	    shape = tt.getShape();
	    for (unsigned i: shape) {
	      if (i == -1)
		dyn_shape = true;
	    }
	    eltType = tt.getElementType();
	  }
	  else {
	    assert(false);
	  }

	  // Value of the cells in the future tensor
	  Attribute constAtt = undefConstOf(eltType);

	  TensorType tensorType  = RankedTensorType::get(shape,eltType);
	  Value cv;
	  if (dyn_shape) {
	    std::vector<int64_t> nshape;
	    for (unsigned i: shape) {
	      if (i == -1)
		nshape.push_back(1);
	      else
		nshape.push_back(i);
	    }
	    TensorType ntensorType  = RankedTensorType::get(nshape,eltType);
	    Attribute denseAtt = DenseElementsAttr::get(ntensorType,constAtt);
	    arith::ConstantOp co = b.create<arith::ConstantOp>(loc,
							       denseAtt);
	    tensor::CastOp op = b.create<tensor::CastOp>(loc,tensorType,
							 co.getResult());
	    cv = op.getResult();
	  }
	  else {
	    Attribute denseAtt = DenseElementsAttr::get(tensorType,constAtt);
	    arith::ConstantOp co = b.create<arith::ConstantOp>(loc,
							       denseAtt);
	    cv = co.getResult();
	  }
		 
	  // Value r;
	  // if (t.dyn_cast<TensorType>()) {
	  //   r = cv;
	  // }
	  // else {
	  //   // Create i = 0: index
	  //   Attribute az = IntegerAttr::get(b.getIndexType(), 0);
	  //   arith::ConstantOp oz = b.create
	  //     <arith::ConstantOp>(loc, az);
	  //   Value z = oz.getResult();
	  //   // Extract t[0]
	  //   tensor::ExtractOp extractOp = b.create
	  //     <tensor::ExtractOp>(loc, cv, z);
	  //   r = extractOp.getResult();
	  // }
	  u.getResult().replaceAllUsesWith(cv);
	  u.erase();

		 
		 // Value r;
		 // if (auto tt = t.dyn_cast<TensorType>()) {
		 //   ArrayRef<int64_t> shape = tt.getShape();
		 //   Type eltType = tt.getElementType();
		 //   linalg::InitTensorOp initTensorOp = b.create
		 //     <linalg::InitTensorOp>(loc,shape,eltType);
		 //   r = initTensorOp.getResult();
		 // }
		 // else if (!t.isa<ShapedType>()) {
		 //   // Create t: tensor<1xt>
		 //   int64_t shape[1] = {1};
		 //   linalg::InitTensorOp initTensorOp = b.create
		 //     <linalg::InitTensorOp>(loc,shape,t);
		 //   Value t = initTensorOp.getResult();

		 //   // Create i = 0: index
		 //   Attribute az = IntegerAttr::get(b.getIndexType(), 0);
		 //   arith::ConstantOp oz = b.create
		 //     <arith::ConstantOp>(loc, az);
		 //   Value z = oz.getResult();

		 //   // Extract t[0]
		 //   tensor::ExtractOp extractOp = b.create
		 //     <tensor::ExtractOp>(loc, t, z);
		 //   r = extractOp.getResult();
		 // }
		 // u.getResult().replaceAllUsesWith(r);
		 // u.erase();


	});
    }

    std::unique_ptr<Pass> createSyncToStandardPass() {
      return std::make_unique<SyncToStandardPass>();
    }
    
  }
}
