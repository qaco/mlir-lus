#include "llvm/ADT/MapVector.h"
#include "ClassicClocker.h"
#include "mlir/IR/AsmState.h"
#include "../../Sync/SyncOp.h"
#include "../../Sync/TickOp.h"
#include "../../../Transforms/Utilities/Helpers.h"

// clang++-9 -fno-rtti -fvisibility-inlines-hidden -Wall -I /home/hpompougnac/llvm/include -I./include -c ClassicClocker.cpp

namespace mlir {
  namespace lus {
    
    // ClockedOutputsHandler
    
    Value ClockedOutputsHandler::concretize(Value a) {
      for (auto e: llvm::zip(abst,conc)) {
	Value abstract = get<0>(e);
	Value concrete = get<1>(e);
	if (a == abstract)
	  return concrete;
      }
      return a;
    }

    // ClassicClocker

    void ClassicClocker::bindSignature() {

      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stderr", err);

      for (Value s: node.getState()) {
	  ClassicClock cc(ClassicClock::PREFIX_BASE);
	  classicEnv.bindValue(s,cc);
      }
      
      if (!node.isClocked()) {
	for (Value inp: node.getInputs()) {
	  ClassicClock cc(ClassicClock::PREFIX_FREE);
	  classicEnv.bindValue(inp,cc);
	}
	if (node.outputsExplicit()) {
	  for (Value outp: node.getOutputs()) {
	    ClassicClock cc(ClassicClock::PREFIX_FREE);
	    classicEnv.bindValue(outp,cc);
	  }
	}
      }
      else {
	SmallVector<ClassicClock,4> clockedInputs;
	node.clockedInputs(clockedInputs);
	SmallVector<ClassicClock,4> clockedOutputs;
	node.clockedOutputs(clockedOutputs);
	if (ClassicClocker::DEBUG_MODE) {
	  OnClockNodeOp signatureClock = node.getSignatureClock();
	  stream << "===CLOCK DESIGN\n";
	  stream << "Signature clock: " << signatureClock << "\n";
	  stream << "Num inputs: "
		 << node.getNumInputs() << "\n";
	  stream << "Num input clocks: "
		 << clockedInputs.size() << "\n";
	  stream << "Num outputs: "
		 << node.getNumOutputs() << "\n";
	  stream << "Num output clocks: "
		 << clockedOutputs.size() << "\n";
	  stream << "===\n";
	}
      
	for (auto e: llvm::zip(node.getInputs(),clockedInputs)) {
	  Value myInput = get<0>(e);
	  ClassicClock cc = get<1>(e);
	  classicEnv.bindValue(myInput,cc);
	}

	// ClockedOutputsHandler h(node.getOutputs(),
	// 			node.getYield().getOutputs());
      
	// SmallVector<Value,4> abstractions;
	// SmallVector<Value,4> concretizations;
	// for (auto abstract_o: node.getOutputs()) {
	//   Value o = h.concretize(abstract_o);
	//   abstractions.push_back(abstract_o);
	//   concretizations.push_back(o);
	// }
	// for (auto cc: clockedOutputs) {
	//   for (auto e: llvm::zip(abstractions,concretizations)) {
	//     Value a = get<0>(e);
	//     Value c = get<1>(e);
	//     cc.replace(a,c);
	//   }
	// }
	for (auto e: llvm::zip(node.getOutputs(),clockedOutputs)) {
	  Value o = get<0>(e);
	  ClassicClock cc = get<1>(e);
	  classicEnv.bindValue(o,cc);
	}
      }
    }

    void ClassicClocker::bindOperation(Operation *op) {
      ClassicClock cc;
      if (isa<sync::SyncOp>(op) || isa<sync::TickOp>(op))
	cc.setPrefix(ClassicClock::PREFIX_BASE);
      else
	cc.setPrefix(ClassicClock::PREFIX_FREE);
      for (Value v: op->getResults()) {
	classicEnv.bindValue(v,cc);
      }
    }
    
    void ClassicClocker::bindBody() {
      for (Operation &o: node.getBody().front().getOperations()) {
	bindOperation(&o);
	if (OnClockOp onClock = dyn_cast<OnClockOp>(&o)) {
	  bindOperation(onClock.nested());
	}
      }
    }

    bool ClassicClocker::unifyClocks(ClassicClock cc1, ClassicClock cc2) {

      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stderr", err);
      
      ClassicClock rcc;
      
      if (cc1.isFree()) {
	rcc = cc2;
      }
      else if (cc2.isFree()) {
	rcc = cc1;
      }
      else if(cc1.isBase() && cc2.isBase()) {
	rcc = cc1;
      }
      else if (cc1.getLength() == cc2.getLength()) {
	for (auto e: llvm::zip(cc1.getSubsamplings(),
			       cc2.getSubsamplings())) {
	  Subsampling s0 = get<0>(e);
	  Subsampling s1 = get<1>(e);
	  if (s0.on_or_onnot != s1.on_or_onnot) {
	    stream << "Inequality (boolean) between "
		   << s0.to_string()
		   << " and " << s1.to_string() << "\n";
	    return false;
	  }
	  if (s0.data != s1.data) {
	    stream << classicEnv.to_string() << "\n";
	    stream << "Inequality (data) between "
	    	   << s0.to_string()
	    	   << " and " << s1.to_string() << "\n";
	    return false;
	  }
	}
	if (cc1.getPrefix() == ClassicClock::PREFIX_FREE) {
	  rcc = cc2;
	}
	else if (cc2.getPrefix() == ClassicClock::PREFIX_FREE) {
	  rcc = cc1;
	}
	else if (cc2.getPrefix() == ClassicClock::PREFIX_BASE
		 && cc1.getPrefix() == cc2.getPrefix()) {
	  rcc = cc1;
	}
	else if (cc2.getPrefix() == ClassicClock::PREFIX_BASE
		 || cc1.getPrefix() == ClassicClock::PREFIX_BASE) {
	  // Todo : base & ptr
	  assert(false);
	}
	else {
	  ClassicClock p0 = classicEnv.getClock(cc1.getPrefix());
	  ClassicClock p1 = classicEnv.getClock(cc2.getPrefix());
	  if (!verboseUnifyClocks(p0,p1))
	    return false;
	}
	rcc = cc1;
      }
      else {
	
	ClassicClock *shorter;
	ClassicClock *longer;
	if (cc1.getLength() > cc2.getLength()) {
	  shorter = &cc2;
	  longer = &cc1;
	}
	else {
	  shorter = &cc1;
	  longer = &cc2;
	}

	if (shorter->getPrefix() == ClassicClock::PREFIX_BASE)
	  return false;
	  
	const unsigned offset = longer->getLength() - shorter->getLength();
	for (unsigned i = 0; i < shorter->getLength(); i++) {
	  Subsampling ss = shorter->getSubsamplings()[i];
	  Subsampling ls = longer->getSubsamplings()[i + offset];
	  if (ss.on_or_onnot != ls.on_or_onnot
	      || ss.data != ls.data)
	    return false;
	}
	  
	SmallVector<Subsampling,4> longerSsLeft;
	for (unsigned i = 0; i < offset; i++) {
	  longerSsLeft.push_back(longer->getSubsamplings()[i]);
	}
	ClassicClock lLeft = classicEnv.generateClock(longer->getPrefix(),
						      longerSsLeft);
	ClassicClock sLeft = classicEnv.getClock(shorter->getPrefix());
	if (!verboseUnifyClocks(lLeft,sLeft))
	  return false;
	rcc = *longer;
      }

      classicEnv.eraseClock(cc1,rcc);
      classicEnv.eraseClock(cc2,rcc);
	
      return true;
    }

    bool ClassicClocker::verboseUnifyClocks(ClassicClock cc1,
					    ClassicClock cc2) {
      if(!unifyClocks(cc1,cc2)) {
	std::error_code err;
	llvm::raw_fd_ostream stream("/dev/stderr", err);
	stream << "Clock error\nUnification impossible between:\n"
	       << "-> " << classicEnv.string_of(cc1) << "\n"
	       << "-> " << classicEnv.string_of(cc2) << "\n";
	return false;
      }
      return true;
    }
    
    ClassicClock ClassicClocker::incrementClock(ClassicClock initial,
						bool onFlag,
						Value cond) {
      Subsampling back(onFlag,cond);
      SmallVector<Subsampling,4> inferredSs;
      for (auto s: initial.getSubsamplings())
	inferredSs.push_back(s);
      inferredSs.push_back(back);
      ClassicClock nc = classicEnv.generateClock(initial.getPrefix(),
						 inferredSs);
      return nc;
    }

    bool ClassicClocker::inferOp(Operation* op) {
      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stderr", err);
      if (WhenOp when = dyn_cast<WhenOp>(op)) {
	return inferWhen(when);
      }
      else if (MergeOp merge = dyn_cast<MergeOp>(op)) {
	return inferMerge(merge);
      }
      else if (OnClockOp onClock = dyn_cast<OnClockOp>(op)) {
	return inferOnClock(onClock);
      }
      else if (InstanceTestOp inst = dyn_cast<InstanceTestOp>(op)) {
	return inferInst(inst);
      }
      else if (!isa<YieldOp>(op)
	       && !isa<sync::SyncOp>(op) && !isa<sync::TickOp>(op)) {
	return inferSimple(op);
      }
      return true;
    }

    bool ClassicClocker::verboseInferOp(Operation *op) {
      if(!inferOp(op)) {
	std::error_code err;
	llvm::raw_fd_ostream stream("/dev/stderr", err);
	stream << "in " << *op << "\n";
	stream << "current env:\n" << classicEnv.to_string();
	return false;
      }
      return true;
    }

    bool ClassicClocker::inferWhen(WhenOp when) {
      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stderr", err);
      Value output = when.getResult();
      Value coming = when.getDataInput();
      Value cond = when.getCondValue();
      bool whenotFlag = when.getCondType().getWhenotFlag();
      ClassicClock comingClock = classicEnv.getClockOfValue(coming);
      ClassicClock outputClock = classicEnv.getClockOfValue(output);
      ClassicClock condClock = classicEnv.getClockOfValue(cond);

      if(!verboseUnifyClocks(comingClock,condClock)) {
	stream << "On values ";
	ClassicClock::printValue(coming,stream);
	stream << " and ";
	ClassicClock::printValue(cond,stream);
	stream << "\n";
	return false;
      }
      ClassicClock unifiedClock = classicEnv.getClockOfValue(coming);

      ClassicClock inferredClock = incrementClock(unifiedClock,
						  !whenotFlag,
						  cond);
      if(!verboseUnifyClocks(outputClock,inferredClock)) {
	stream << "Value ";
	ClassicClock::printValue(output,stream);
	stream << " must be clocked on\n"
	       << "-> " << inferredClock.to_string() << "\n"
	       << "instead of\n-> " << outputClock.to_string() << "\n";
	return false;
      }
      return true;
    }

    bool ClassicClocker::inferMerge(MergeOp merge) {
      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stderr", err);
      Value output = merge.getResult();
      Value trueComing = merge.getTrueInput();
      Value falseComing = merge.getFalseInput();
      Value cond = merge.getCondValue();
      ClassicClock trueComingClock =
	classicEnv.getClockOfValue(trueComing);
      ClassicClock falseComingClock =
	classicEnv.getClockOfValue(falseComing);
      ClassicClock outputClock = classicEnv.getClockOfValue(output);
      ClassicClock condClock = classicEnv.getClockOfValue(cond);

      if(!verboseUnifyClocks(outputClock,condClock)) {
	stream << "On values ";
	ClassicClock::printValue(output,stream);
	stream << " and ";
	ClassicClock::printValue(cond,stream);
	stream << "\n";
	return false;
      }
      ClassicClock unifiedClock = classicEnv.getClockOfValue(output);

      ClassicClock inferredTrue = incrementClock(unifiedClock,true,cond);
      if(!verboseUnifyClocks(trueComingClock,inferredTrue)) {
	stream << "On value " << trueComing;
	// ClassicClock::printValue(trueComing,stream);
	stream << " (true branch)\n";
	return false;
      }

      ClassicClock inferredFalse = incrementClock(unifiedClock,
						  false,cond);
      if(!verboseUnifyClocks(falseComingClock,inferredFalse)) {
	stream << "On value ";
	ClassicClock::printValue(falseComing,stream);
	stream << " (false branch)\n";
	return false;
      }
      return true;
    }

    
    bool ClassicClocker::inferOnClock(OnClockOp onClock) {
      Operation *nested = &onClock.getBody().front().front();
      if (!verboseInferOp(nested))
	return false;

      SmallVector<ClassicClock,4> outputClocks;
      onClock.clockedOutputs(outputClocks);
      for (auto e : llvm::zip(onClock.getResults(),
			      nested->getResults(),
			      outputClocks)) {
	Value oExt = get<0>(e);
	ClassicClock occ = classicEnv.getClockOfValue(oExt);
	Value iExt = get<1>(e);
	ClassicClock icc = classicEnv.getClockOfValue(iExt);
	ClassicClock cca = get<2>(e);
	classicEnv.rememberClock(cca);
	if(!verboseUnifyClocks(occ,cca)) {
	  return false;
	}
	if(!verboseUnifyClocks(icc,cca)) {
	  return false;
	}
      }
      return true;
    }

    bool ClassicClocker::inferInst(InstanceTestOp instanceOp) {

      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stdout", err);

      NodeTestOp callee = instanceOp.getCalleeNode();

      if (!callee.isClocked()) {
	return inferSimple(instanceOp);
      }
      
      SmallVector<Value,4> calleeIos;
      calleeIos.append(callee.getInputs().begin(),
		       callee.getInputs().end());
      calleeIos.append(callee.getOutputs().begin(),
		       callee.getOutputs().end());
      SmallVector<Value,4> instIos;
      instIos.append(instanceOp.getArgOperands().begin(),
		     instanceOp.getArgOperands().end());
      instIos.append(instanceOp.getResults().begin(),
		     instanceOp.getResults().end());
      llvm::SmallMapVector<Value,Value,4> replacements;
      for (auto e: llvm::zip(calleeIos, instIos)) {
	Value vNode = get<0>(e);
	Value vInst = get<1>(e);
	replacements[vNode] = vInst;
      }
      SmallVector<ClassicClock,4> calleeInputClocks;
      callee.clockedInputs(calleeInputClocks);
      SmallVector<ClassicClock,4> calleeOutputClocks;
      callee.clockedOutputs(calleeOutputClocks);
      SmallVector<ClassicClock,4> calleeIosClocks;
      calleeIosClocks.append(calleeInputClocks.begin(),
			     calleeInputClocks.end());
      calleeIosClocks.append(calleeOutputClocks.begin(),
			     calleeOutputClocks.end());
      SmallVector<ClassicClock,4> instIosClocks;
      for (auto v: instIos) {
	instIosClocks.push_back(classicEnv.getClockOfValue(v));
      }
      for (auto e: llvm::zip(calleeIosClocks, instIosClocks)) {
	ClassicClock ccCallee = get<0>(e);
	ClassicClock ccInst = get<1>(e);
	for (Subsampling &s: ccCallee.getSubsamplings()) {
	  s.data = replacements[s.data];
	}
	classicEnv.rememberClock(ccCallee);
	if (!verboseUnifyClocks(ccCallee,ccInst)) {
	  return false;
	}
      }
      return true;
    }
    
    bool ClassicClocker::inferSimple(Operation* op) {
      SmallVector<Value> ios;
      ios.append(op->getResults().begin(),
		 op->getResults().end());
      ios.append(op->getOperands().begin(),
		 op->getOperands().end());
      if (ios.size() > 1) {
	Value front = ios[0];
	for (unsigned i = 1; i < ios.size(); i++) {
	  Value tmp = ios[i];
	  ClassicClock frontClock = classicEnv.getClockOfValue(front);
	  ClassicClock tmpClock = classicEnv.getClockOfValue(tmp);
	  if(!verboseUnifyClocks(frontClock,tmpClock))
	    return false;
	}
      }
      return true;
    }

    ClassicClockEnv ClassicClocker::infer() {
      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stderr", err);
      bindSignature();
      bindBody();
      for (Operation &op: node.getBody().front().getOperations()) {
	bool res = verboseInferOp(&op);
	if (!res) {
	  stream << "(node " << node.sym_name() << ")\n";
	  assert(false);
	}
      }
      classicEnv.putAllOnBaseClock();
      assert(classicEnv.isClockingCausal());
      return classicEnv;
    }
  }
}
