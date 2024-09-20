// -*- C++ -*- //

#ifndef VARIANT_TYPE_H
#define VARIANT_TYPE_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
  namespace mytf {

    class VariantTypeStorage: public TypeStorage {
    private:
      Type tensorTy;
    public:

      const Type getType() const { return tensorTy; }
      
      // Uniquing
      using KeyTy = Type;
      bool operator==(const KeyTy &key) const {
	return key == getType();
      }

      // Construction

      VariantTypeStorage(const Type& tTy) : TypeStorage(), tensorTy(tTy) {
	assert(tTy.isa<TensorType>());
      }

      static VariantTypeStorage *construct(TypeStorageAllocator &allocator,
					    const KeyTy &key) {
	return new (allocator.allocate<VariantTypeStorage>())
	  VariantTypeStorage(key);
      }
    };

    class VariantType: public Type::TypeBase<VariantType,
					      Type,
					      VariantTypeStorage> {
    public:
      using Base::Base;

      static VariantType get(MLIRContext *context, Type tTy) {
	assert(tTy.isa<TensorType>());
	return Base::get(context, tTy);
      }

      const Type getType() const {
	return getImpl()->getType(); }

    };
  }
}

#endif
