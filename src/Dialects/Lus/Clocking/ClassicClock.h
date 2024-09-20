// -*- C++ -*- //

#ifndef CLASSIC_CLOCK_H
#define CLASSIC_CLOCK_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
  namespace lus {

    struct Subsampling {
    private:
    public:
      Subsampling() = delete;
      Subsampling(bool b, Value v): on_or_onnot(b), data(v) {}
      Subsampling(const Subsampling &s): on_or_onnot(s.on_or_onnot),
					 data(s.data) {}
      Subsampling(Subsampling &s): on_or_onnot(s.on_or_onnot),
				   data(s.data) {}
      bool on_or_onnot;
      Value data;
      std::string to_string();
    };
    
    class ClassicClock {
      
    private:

      static const bool DEBUG_MODE = false;
      static unsigned id_count;
      unsigned unique_id;
      int prefix;
      SmallVector<Subsampling,4> subsamplings;
      
    public:
      static void printValue(Value v,raw_ostream &stream);
      static const int PREFIX_BASE = -1;
      static const int PREFIX_FREE = -2;
      ClassicClock();
      ClassicClock(unsigned p);
      ClassicClock(SmallVectorImpl<Subsampling> &v);
      ClassicClock(unsigned p, SmallVectorImpl<Subsampling> &v);
      bool isFree() const {
	return prefix == PREFIX_FREE && subsamplings.empty();
      }
      bool isBase() const {
	return prefix == PREFIX_BASE && subsamplings.empty();
      }
      unsigned getLength() { return subsamplings.size(); }
      SmallVectorImpl<Subsampling>& getSubsamplings() { return subsamplings;}
      int getPrefix() { return prefix; }
      void setPrefix(int p) { prefix = p; }
      void setSubsamplings(SmallVectorImpl<Subsampling> &s) {
	subsamplings.clear();
	subsamplings.append(s.begin(),s.end());
      }
      void replace(Value oldv, Value newv);
      std::string to_string() const;
      ClassicClock copy();
      bool equals(const ClassicClock& cc) const {
	return (unique_id == cc.unique_id
		|| (isBase() && cc.isBase()));
      }
      bool operator==(const ClassicClock& cc) const {
	  return this->equals(cc);
	}
    };

    class ClassicClockEnv {
      
    private:
      
      static const bool DEBUG_MODE = false;
      static const unsigned INDEX_OF_BASE = 0;
      SmallVector<Value,4> values;
      SmallVector<unsigned,4> clocks_of_values;
      SmallVector<ClassicClock,4> clocks;
      
      unsigned indexOfClock(const ClassicClock& cc) const;
      unsigned indexOfValue(Value& v) const;
      void putOnBaseClock(unsigned index);
      bool valueAppearsInClock(Value v, ClassicClock c);

    public:
      ClassicClock getBaseClock() const { return clocks[INDEX_OF_BASE] ; }
      void rememberClock(ClassicClock cc) { clocks.push_back(cc); }
      ClassicClock getClock(unsigned i) { return clocks[i]; }
      bool isValueBound(const Value& v) const;
      bool isClockBound(const ClassicClock& cc) const;
      ClassicClock generateClock(unsigned prefix,
				 SmallVectorImpl<Subsampling> &ss);
      void bindValue(Value& v, ClassicClock cc);
      void bindValue(Value& v);
      ClassicClock getClockOfValue(Value& v) const;
      ClassicClock getClockOfOp(Operation *op) const;
      void putValueOnClock(Value& v, const ClassicClock& cc);
      void eraseClock(const ClassicClock& oldClock,
		      const ClassicClock& newClock);
      void putAllOnBaseClock();
      bool isClockingCausal();
      std::string string_of(const ClassicClock& c);
      std::string to_string() const;
      ClassicClockEnv() {
	ClassicClock cc(ClassicClock::PREFIX_BASE);
	rememberClock(cc);
      }
    };

  }
}

#endif
