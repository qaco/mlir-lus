// -*- C++ -*- //

#ifndef RESOURCE_TYPE_H
#define RESOURCE_TYPE_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
  namespace mytf {

    class ResourceTypeStorage: public TypeStorage {
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

      ResourceTypeStorage(const Type& tTy) : TypeStorage(), tensorTy(tTy) {
	assert(tTy.isa<TensorType>());
      }

      static ResourceTypeStorage *construct(TypeStorageAllocator &allocator,
					    const KeyTy &key) {
	return new (allocator.allocate<ResourceTypeStorage>())
	  ResourceTypeStorage(key);
      }
    };

    class ResourceType: public Type::TypeBase<ResourceType,
					      Type,
					      ResourceTypeStorage> {
    public:
      using Base::Base;

      static ResourceType get(MLIRContext *context, Type tTy) {
	assert(tTy.isa<TensorType>());
	return Base::get(context, tTy);
      }

      const Type getType() const {
	return getImpl()->getType(); }

    };
  }
}

#endif
