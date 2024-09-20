#ifndef CLOCK_TYPES_H
#define CLOCK_TYPES_H

#include <tuple>
#include "mlir/IR/Types.h"
#include "TestCondition.h"
#include "ClockTree.h"
#include <bitset>

namespace mlir {
  namespace lus {

    class ClockTypeStorage : public TypeStorage {
    private:
      unsigned long seq;
      size_t size;
            
    public:
      const std::vector<bool> getSeq() const {
	std::bitset<64> bitset(seq);
	std::vector<bool> ret;
	for (size_t i = 0; i < size; i++)
	  ret.push_back(bitset[i]);
	return ret;
      }
      
    public:
      using KeyTy = std::pair<unsigned long,size_t>;
      bool operator==(const KeyTy &key) const {
	return key == KeyTy(seq,size) ;
      }
      
	ClockTypeStorage(const unsigned long seq,
			 const size_t size)
	: TypeStorage(),
	  seq(seq),
	  size(size)
      {}

      /// Construction.
      
      static ClockTypeStorage *construct(TypeStorageAllocator &allocator,
					 const KeyTy &key) {

	const unsigned long seq = std::get<0>(key);
	const size_t size = std::get<1>(key);
	
	return new (allocator.allocate<ClockTypeStorage>())
	  ClockTypeStorage(seq, size);
      }      
    };

    ///---------------------------------------------------------
    class ClockType : public Type::TypeBase<ClockType,
					   Type,
					   ClockTypeStorage> {
    public:
      using Base::Base;
      
      static ClockType get(MLIRContext *context,
			   const std::vector<bool> flags) {
	size_t size = flags.size();
	assert(size < 64);
	std::bitset<64> bits;
	for (size_t i = 0; i < size; i++) {
	  bits[i] = flags[i];
	}
	unsigned long seq = bits.to_ulong();
	return Base::get(context,seq,size);
      }
      
      // Accessors
      const std::vector<bool> getSeq() const  { return getImpl()->getSeq(); }
    };
  }
}

#endif
