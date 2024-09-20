#include "ClassicClock.h"
#include "mlir/IR/AsmState.h"
#include "../../Sync/OutputOp.h"
#include "../OnClockOp.h"
#include "../WhenOp.h"
#include "../MergeOp.h"
#include "../InstanceTest.h"

namespace mlir {
  namespace lus {

    std::string Subsampling::to_string() {
      std::string ret;
      llvm::raw_string_ostream stream(ret);
      stream << "(on ";
      if (!on_or_onnot)
	stream << "not ";
      stream << "(" << data << ")";
      // ClassicClock::printValue(data,stream);
      // stream << ")";
      return ret;
    }
    
    void ClassicClock::printValue(Value v,raw_ostream &stream) {
      if (v.isa<BlockArgument>()) {
	BlockArgument ba = v.cast<BlockArgument>();
	AsmState state(ba.getOwner()->getParentOp());
	v.printAsOperand(stream,state);
      }
      else {
	AsmState state(v.getDefiningOp());
	v.printAsOperand(stream,state);
      }
    }
    
    // ClassicClock

    unsigned ClassicClock::id_count = 0;
    
    ClassicClock::ClassicClock() {
      prefix = PREFIX_FREE;
      unique_id = id_count++;
    }

    ClassicClock::ClassicClock(unsigned p) {
      prefix = p;
      unique_id = id_count++;
    }

    ClassicClock::ClassicClock(SmallVectorImpl<Subsampling> &v) {
      prefix = PREFIX_FREE;
      unique_id = id_count++;
      for (Subsampling s: v)
	subsamplings.push_back(s);
    }

    ClassicClock::ClassicClock(unsigned p, SmallVectorImpl<Subsampling> &v) {
      prefix = p;
      unique_id = id_count++;
      for (Subsampling s: v)
	subsamplings.push_back(s);
    }
    
    void ClassicClock::replace(Value oldv, Value newv) {
      std::error_code err;
      llvm::raw_fd_ostream errstream("/dev/stdout", err);

      if (ClassicClock::DEBUG_MODE) {
	errstream << "===REPLACEMENT\n"
		  << "-> in " << to_string() << "\n"
		  << "-> old: ";
	ClassicClock::printValue(oldv,errstream);
	errstream << "\n-> new: ";
	ClassicClock::printValue(newv,errstream);
	errstream << "\n===\n";
      }
      
      for (unsigned i = 0; i < subsamplings.size(); i++) {
	if (subsamplings[i].data == oldv) {
	  subsamplings[i].data = newv;

	  if (ClassicClock::DEBUG_MODE) {
	    errstream << "===REPLACEMENT DONE\n"
		      << "old: ";
	    ClassicClock::printValue(oldv,errstream);
	    errstream << "\nnew: ";
	    ClassicClock::printValue(newv,errstream);
	    errstream << "\n===\n";
	  }
	}
	  
      }
    }

    std::string ClassicClock::to_string() const {
      std::error_code err;
      llvm::raw_fd_ostream errstream("/dev/stdout", err);
      
      std::string ret;
      llvm::raw_string_ostream stream(ret);

      stream << "(id=" << std::to_string(unique_id) << ") ";

      if (prefix == PREFIX_BASE)
	stream << "base";
      else if (prefix == PREFIX_FREE)
	stream << "free";
      else
	stream << "clock" << std::to_string(prefix);

      for (Subsampling s: subsamplings) {
	stream << " on ";
	if (!s.on_or_onnot)
	  stream << "not ";
	stream << "(" << s.data << ")";
	// ClassicClock::printValue(s.data,stream);
      }
      return ret;
    }

    ClassicClock ClassicClock::copy() {
      int prefix;
      if (getPrefix() == PREFIX_BASE || getPrefix() == PREFIX_FREE)
	prefix = getPrefix();
      else
	prefix = id_count++;
      SmallVector<Subsampling,4> subsamplings;
      for (Subsampling s: getSubsamplings()) {
	Subsampling n(s.on_or_onnot, s.data);
	subsamplings.push_back(s);
      }
      ClassicClock cpy(prefix,subsamplings);
      return cpy;
    }

    // ClassicClockEnv

    unsigned ClassicClockEnv::indexOfClock(const ClassicClock& cc) const {
      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stderr", err);
      for (unsigned i = 0; i < clocks.size(); i++) {
	if (clocks[i] == cc)
	  return i;
      }
      stream << "FATAL ERROR\n"
	     << "Clock [" << cc.to_string()
	     << "] is not bound in the clock environment.\n";
      assert(false);
    }

    unsigned ClassicClockEnv::indexOfValue(Value& v) const {
      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stderr", err);
      for (unsigned i = 0; i < values.size(); i++) {
	if (values[i] == v)
	  return i;
      }
      stream << "FATAL ERROR\n";
      stream << "Value " << v;
      stream << " is not bound in the clock environment.\n";
      stream << "Bounded values:\n";
      for (Value b: values) {
	stream << b << "\n";
      }
      assert(false);
    }

    bool ClassicClockEnv::isValueBound(const Value& v) const {
      for (unsigned i = 0; i < values.size(); i++) {
	if (values[i] == v)
	  return true;
      }
      return false;
    }

    bool ClassicClockEnv::isClockBound(const ClassicClock& cc) const {
      for (unsigned i = 0; i < clocks.size(); i++) {
	if (clocks[i] == cc)
	  return true;
      }
      return false;
    }

    ClassicClock ClassicClockEnv::generateClock(unsigned prefix,
						SmallVectorImpl<Subsampling> &ss) {
      ClassicClock cc(prefix,ss);
      clocks.push_back(cc);
      return cc;
    }
      
    void ClassicClockEnv::bindValue(Value& v, ClassicClock cc) {
      if (isValueBound(v))
	return;
      // if (isValueBound(v) || isClockBound(cc))
      //   assert(false);
      clocks.push_back(cc);
      values.push_back(v);
      clocks_of_values.push_back(clocks.size() - 1);
      
      if (ClassicClockEnv::DEBUG_MODE) {
	std::error_code err;
	llvm::raw_fd_ostream stream("/dev/stderr", err);
	stream << "===BINDING\n"
	       << "Value: " << v;
	stream << "\nClock: " << cc.to_string()
	       << "\n===\n";
      }
    }
    

    void ClassicClockEnv::bindValue(Value& v) {
      ClassicClock cc;
      bindValue(v,cc);
    }

    ClassicClock ClassicClockEnv::getClockOfValue(Value& v) const {
      const unsigned value_index = indexOfValue(v);
      const unsigned clock_index = clocks_of_values[value_index];
      return  clocks[clock_index];
    }

    ClassicClock ClassicClockEnv::getClockOfOp(Operation *op) const {
      
      if (OnClockOp onClockOp = dyn_cast<OnClockOp>(op)) {
	return onClockOp.eqClock();
      }
      else if (OnClockOp onClockOp = dyn_cast<OnClockOp>(op->getParentOp())) {
	return onClockOp.eqClock();
      }
      else if (WhenOp whenOp = dyn_cast<WhenOp>(op)) {
	Value v = whenOp.getDataInput();
	return getClockOfValue(v);
      }
      else if (MergeOp mergeOp = dyn_cast<MergeOp>(op)) {
	Value v = mergeOp.getResult();
	return getClockOfValue(v);
      }
      else if (sync::OutputOp outputOp = dyn_cast<sync::OutputOp>(op)) {
	Value v = outputOp.getParameter();
	return getClockOfValue(v);
      }
      else if (InstanceTestOp instOp = dyn_cast<InstanceTestOp>(op)) {
	SmallVector<Value,4> ios;
	ios.append(instOp.getResults().begin(),
		   instOp.getResults().end());
	ios.append(instOp.getArgOperands().begin(),
		   instOp.getArgOperands().end());

	if (ios.size() == 1) {
	  return getClockOfValue(ios[0]);
	}

	// If signature heterogeneous : base clock
	ClassicClock cc1 = getClockOfValue(ios[0]);
	for (unsigned i = 1; i < ios.size(); i++) {
	  ClassicClock cc2 = getClockOfValue(ios[i]);
	  if (!cc1.equals(cc2))
	    return getBaseClock();
	  cc1 = cc2;
	}
      }
	
      // Generic (works for SyncOp and TickOp too)
      for (Value v: op->getResults())
	return getClockOfValue(v);
      for (Value v: op->getOperands())
	return getClockOfValue(v);

      return getBaseClock();
    }

    void ClassicClockEnv::putValueOnClock(Value& v,
					  const ClassicClock& cc) {
      const unsigned value_index = indexOfValue(v);
      const unsigned clock_index = indexOfClock(cc);
      clocks_of_values[value_index] = clock_index;
    }
    

    void ClassicClockEnv::eraseClock(const ClassicClock& oldClock,
				     const ClassicClock& newClock) {
      if (oldClock == newClock)
	return;
      const unsigned oldclock_index = indexOfClock(oldClock);
      const unsigned newclock_index = indexOfClock(newClock);
      for (unsigned i = 0; i < clocks_of_values.size(); i++) {
	if (clocks_of_values[i] == oldclock_index) {
	  clocks_of_values[i] = newclock_index;
	}
      }
    }

    void ClassicClockEnv::putAllOnBaseClock() {
      for (unsigned c: clocks_of_values) {
	putOnBaseClock(c);
      }
    }

    void ClassicClockEnv::putOnBaseClock(const unsigned index) {
      ClassicClock cc = clocks[index];
      const unsigned prefix = cc.getPrefix();
      
      if (prefix == ClassicClock::PREFIX_BASE) {
	return;
      }
      else if (prefix == ClassicClock::PREFIX_FREE) {
	clocks[index].setPrefix(ClassicClock::PREFIX_BASE);
      }
      else {
	putOnBaseClock(prefix);
	ClassicClock prefixClock = getClock(prefix);
	SmallVector<Subsampling,4> ss;
	ss.append(prefixClock.getSubsamplings().begin(),
		  prefixClock.getSubsamplings().end());
	ss.append(cc.getSubsamplings().begin(),
		  cc.getSubsamplings().end());
	clocks[index].setSubsamplings(ss);
	clocks[index].setPrefix(ClassicClock::PREFIX_BASE);
      }
    }

    bool ClassicClockEnv::valueAppearsInClock(Value v, ClassicClock c) {
      for (auto s: c.getSubsamplings()) {
	if (v == s.data)
	  return true;
	ClassicClock cc = getClockOfValue(s.data);
	if (valueAppearsInClock(v,cc))
	  return true;
      }
      return false;
    }
    
    bool ClassicClockEnv::isClockingCausal() {
      
      std::error_code err;
      llvm::raw_fd_ostream stream("/dev/stderr", err);
      for (unsigned i = 0; i < values.size(); i++) {
	Value v = values[i];
	ClassicClock cc = clocks[clocks_of_values[i]];
	if (ClassicClockEnv::DEBUG_MODE) {
	  stream << "===CAUSALITY\n"
		 << "Value: ";
	  ClassicClock::printValue(v,stream);
	  stream << "\nClock: " << cc.to_string() << "\n===\n";
	}
	if (valueAppearsInClock(v,cc)) {
	  stream << "Self-dependence: ";
	  ClassicClock::printValue(v,stream);
	  stream << " clocked on [" << cc.to_string() << "]\n";
	  return false;
	}
      }
      return true;
    }

    // bool ClassicClockEnv::areIOsSubsampled() {
    //   NodeTestOp n = dyn_cast<NodeTestOp>(nodeOp);
    //   for (Value v: n.getInputs()) {
    // 	ClassicClock cc = getClockOfValue(v);
    // 	if (!cc.isBase())
    // 	  return true;
    //   }
    //   for (Value v: n.getOutputs()) {
    // 	ClassicClock cc = getClockOfValue(v);
    // 	if (!cc.isBase())
    // 	  return true;
    //   }
    //   return false;
    // }

    std::string ClassicClockEnv::to_string() const {
      std::string ret;
      llvm::raw_string_ostream stream(ret);
      for (Value v: values) {
	ClassicClock cc = getClockOfValue(v);
	stream << "* Value: " << v << "\n";
	stream << "  Clock: " << cc.to_string() << "\n";
      }
      return ret;
    }
    
    std::string ClassicClockEnv::string_of(const ClassicClock& c) {
      unsigned current = indexOfClock(c);
      std::string ret = "Clock" + std::to_string(current) + ": ";
      ret += c.to_string();
      return ret;
    }
  }
}
