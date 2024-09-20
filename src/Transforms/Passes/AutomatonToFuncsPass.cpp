#include "llvm/ADT/SmallSet.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/Passes.h"
#include "Passes.h"
#include "../../Dialects/Sync/InstOp.h"
#include "../../Dialects/Sync/InputOp.h"
#include "../../Dialects/Sync/OutputOp.h"

namespace mlir {
  namespace sync {

    struct AutomatonToFuncsPass : public PassWrapper< AutomatonToFuncsPass,
						OperationPass<ModuleOp>> {
      void runOnOperation() override;
    };

    TensorType abstract_tensor(Type t) {
      if (t.isa<TensorType>()) {
	TensorType tt = t.cast<TensorType>();
	return tt;
      }
      else {
	std::vector<int64_t> shape(1);
	shape[0] = -1;
	return RankedTensorType::get(shape, t);
      }
    }

    MemRefType abstract_buffer(Type t) {
      if (auto tt = t.cast<TensorType>()) {
	std::vector<int64_t> shape = tt.getShape();
	for (unsigned i = 0; i < shape.size(); i++) {
	  shape[i] = -1;
	}
	MemRefType mrt = MemRefType::get(shape, tt.getElementType());
	return mrt;
      }
      else {
	std::vector<int64_t> shape(1);
	shape[0] = -1;
	MemRefType mrt = MemRefType::get(shape, t);
	return mrt;
      }
    }

    MemRefType concrete_buffer(Type t) {
      if (auto tt = t.cast<TensorType>()) {
	std::vector<int64_t> shape = tt.getShape();
	MemRefType mrt = MemRefType::get(shape, tt.getElementType());
	return mrt;
      }
      else {
	std::vector<int64_t> shape(1);
	shape[0] = 1;
	MemRefType mrt = MemRefType::get(shape, t);
	return mrt;
      }
    }

    Value allocate(OpBuilder &b, Location &l, Type t) {
      MemRefType concmrty = concrete_buffer(t);
      MemRefType absmrty = abstract_buffer(t);
      memref::AllocOp alloc = b.create<memref::AllocOp>(l,concmrty);
      Value allocated = alloc.getResult();
      memref::CastOp cast = b.create<memref::CastOp>(l,allocated,absmrty);
      return cast.getResult();
    }

    Value bufferize(OpBuilder &b, Location &l, Value v) {
      Type t = v.getType();
      MemRefType concmrty = concrete_buffer(t);
      MemRefType absmrty = abstract_buffer(t);
      if (t.isa<TensorType>()) {
	auto bcast = b.create<memref::BufferCastOp>(l, concmrty,v);
	Value memref = bcast.getResult();
	memref::CastOp cast = b.create<memref::CastOp>(l, memref, absmrty);
	return cast.getResult();
      }
      else {
	Value memref = allocate(b, l, t);
	Attribute zeroAttr = IntegerAttr::get(b.getIndexType(), 0);
	auto zeroOp = b.create<arith::ConstantOp>(l, zeroAttr);
	Value z = zeroOp.getResult();
	b.create<memref::StoreOp>(l, v, memref, z);
	return memref;
      }
    }

    std::string printable(Type t) {
      std::string name;
      llvm::raw_string_ostream stream(name);
      stream << t;
      name.erase(std::remove(name.begin(), name.end(), '<'), name.end());
      name.erase(std::remove(name.begin(), name.end(), '>'), name.end());
      name.erase(std::remove(name.begin(), name.end(), '?'), name.end());
      return name;
    }

    FuncOp get_func(OpBuilder &b,
		    Location &l,
		    std::string name,
		    FunctionType ft,
		    StringAttr visibility,
		    llvm::SmallMapVector<StringRef, FuncOp, 4> &funcs) {
      
      if (auto ff = funcs[name]) {
	return ff;
      }
      else {
	StringAttr fname = StringAttr::get(b.getContext(),
					   name);
	FuncOp f = b.create<FuncOp>(l, fname, ft, visibility);
	funcs[name] = f;
	return f;
      }
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

    Value get_func_ptr(OpBuilder &b,
		       Location &l,
		       std::string name,
		       FunctionType ft,
		       StringAttr visibility,
		       llvm::SmallMapVector<StringRef,FuncOp,4> &funcs) {
       FuncOp f = get_func(b, l, name, ft, visibility, funcs);
       Attribute fname = FlatSymbolRefAttr::get(b.getContext(),f.sym_name());
       ConstantOp fptr = b.create<ConstantOp>(l,f.getType(),fname);
       return fptr.getResult();
    }

    
    
    void AutomatonToFuncsPass::runOnOperation() {

      auto mod = getOperation();
      OpBuilder tlb(mod.getBodyRegion());
      const Type posTy = tlb.getI32Type();
      const Type syncTy = tlb.getI32Type();
      const Type instTy = tlb.getI32Type();
      StringAttr visibility = tlb.getStringAttr("private") ;
      llvm::SmallMapVector<StringRef, FuncOp, 4> funcs;

      llvm::SmallSet<NodeOp,4> instantiatedNodes;
      llvm::SmallSet<InstOp,4> instOps;
      mod.walk([&](InstOp inst) {
	  instOps.insert(inst);
	  auto nodeOp = dyn_cast<NodeOp>(inst.getCalleeNode());
	  instantiatedNodes.insert(nodeOp);
	});

      for (InstOp inst: instOps) {
	OpBuilder instb(inst);
	Location instl = inst.getLoc();
	StringRef nname = inst.getCalleeName();
	//
	// The core function
	//
	
	auto nodeOp = dyn_cast<NodeOp>(inst.getCalleeNode());
	assert(nodeOp);

	// Translate and pack the signal types
	SmallVector<Type, 4> cioTys;
	cioTys.push_back(instTy); // The inst id
	for (Value i : nodeOp.getInputs()) {
	  SiginType st = i.getType().cast<SiginType>();
	  Type t = st.getType();
	  Type tt = abstract_tensor(t);
	  FunctionType ft = instb.getFunctionType({posTy}, {tt});
	  cioTys.push_back(ft);
	}
	for (Value i : nodeOp.getOutputs()) {
	  SigoutType st = i.getType().cast<SigoutType>();
	  Type t = st.getType();
	  Type mrt = abstract_buffer(t);
	  FunctionType ft = instb.getFunctionType({posTy, mrt}, {syncTy});
	  cioTys.push_back(ft);
	}

	// Build the core function
	TypeAttr cft = TypeAttr::get(tlb.getFunctionType(cioTys, {}));
	StringAttr cfname = StringAttr::get(tlb.getContext(),
					    nname);
	FuncOp cf = tlb.create<FuncOp>(instl, cfname, cft, visibility);

	// Region stuff
	nodeOp.getBody().insertArgument((unsigned)0, instTy);
	cf.getBody().takeBody(nodeOp.getBody());
	for (unsigned i = 1 ; i < cioTys.size() ; i++) {
	  Type t = cioTys[i];
	  Value neoArg = cf.getBody().insertArgument(i, t);
	  Value oldArg = cf.getBody().getArgument(i + 1);
	  oldArg.replaceAllUsesWith(neoArg);
	  cf.getBody().eraseArgument(i + 1);
	}

	const unsigned off_out=nodeOp.getNumInputs()+nodeOp.getNumStatic()+1;
	
	// Lower sync::InputOp
	cf.walk([&](InputOp inp) {
	    OpBuilder inpb(inp);
	    Location inpl = inp.getLoc();
	    Value myFun = inp.getSignal();
	    unsigned i = argument_position(myFun, cf, 1);
	    IntegerAttr posAttr = IntegerAttr::get(posTy, i);
	    arith::ConstantOp posop = inpb.create<arith::ConstantOp>(inpl,
								     posAttr);
	    Value pos = posop.getResult();
	    OperationState callState(inpl,CallIndirectOp::getOperationName());
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
	    Value param = bufferize(outpb, outpl, outp.getParameter());
	    OperationState callState(outpl,
				     CallIndirectOp::getOperationName());
	    CallIndirectOp::build(outpb, callState, myFun, {pos, param});
	    Operation *call = outpb.createOperation(callState);
	    outp.replaceAllUsesWith(call);
	    outp.erase();
	  });

	//
	// The start function
	//

	std::string startname = nname.str() + "_start";
	FunctionType startft = instb.getFunctionType({instTy}, {});
	FuncOp sf = tlb.create<FuncOp>(instl, startname, startft, visibility);
	OpBuilder bsf(&sf.getBody());
	
	SmallVector<Type, 4> sioTys;
	for (Type t : inst.getArgOperands().getTypes()) {
	  Type st = SiginType::get(instb.getContext(), t);
	  sioTys.push_back(st);
	}
	for (Type t : inst.getResults().getTypes()) {
	  Type st = SigoutType::get(instb.getContext(), t);
	  sioTys.push_back(st);
	}

	SmallVector<Value, 4> callArgs;
	Block * b = sf.addEntryBlock();
	Value idVal = b->getArgument(0);
	callArgs.push_back(idVal);
	for (Type t : sioTys) {
	  if (auto is = t.cast<sync::SiginType>()) {
	    Type it = abstract_tensor(is.getType());
	    FunctionType ift = instb.getFunctionType({posTy}, {it});
	    Value ifptr = get_func_ptr(tlb, instl,
				       "sched_read_input_" + printable(it),
				       ift, visibility, funcs);
	    callArgs.push_back(ifptr);
	  }
	  else if (auto os = t.cast<sync::SigoutType>()) {
	    Type om = abstract_buffer(os.getType());
	    FunctionType oft = instb.getFunctionType({posTy, om}, {syncTy});
	    Value ofptr = get_func_ptr(tlb, instl,
				       "sched_write_output_" + printable(om),
				       oft, visibility, funcs);
	    callArgs.push_back(ofptr);
	  }
	  else {
	    assert(false);
	  }

	  OperationState callState(instl, CallOp::getOperationName());
	  CallOp::build(bsf, callState, nname, {}, callArgs);
	  bsf.createOperation(callState);
	  bsf.create<ReturnOp>(instl);
	}

	//
	// The inst function
	//

	
      }




      for (NodeOp n: instantiatedNodes) {
	n.erase();
      }
    }

    std::unique_ptr<Pass> createAutomatonToFuncsPass() {
      return std::make_unique<AutomatonToFuncsPass>();
    }
    
  }
}
