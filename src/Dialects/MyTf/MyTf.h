// -*- C++ -*- //

#ifndef DIALECT_MYTF_H
#define DIALECT_MYTF_H

#include "mlir/Transforms/InliningUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
  namespace mytf {

    struct MyTfTypeInlinerInterface : public DialectInlinerInterface {
      using DialectInlinerInterface::DialectInlinerInterface;
      bool isLegalToInline(Operation *, Region *, bool,
			   BlockAndValueMapping &) const final {
	return true;
      }
    };

    class MyTfType : public Dialect {
      public:
      static llvm::StringRef getDialectNamespace() { return "tf_type"; }
      explicit MyTfType(MLIRContext *context) ;

      Type parseType(DialectAsmParser &parser) const override;
      void printType(Type type, DialectAsmPrinter &printer) const override;
    };

    struct MyTfInlinerInterface : public DialectInlinerInterface {
      using DialectInlinerInterface::DialectInlinerInterface;
      bool isLegalToInline(Operation *, Region *, bool,
			   BlockAndValueMapping &) const final {
	return true;
      }
    };

    class MyTf : public Dialect {
      public:
      static llvm::StringRef getDialectNamespace() { return "tf"; }
      explicit MyTf(MLIRContext *context) ;

      void printType(Type type, DialectAsmPrinter &printer) const override {}
    };
  }
}

#endif
