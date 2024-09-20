#include "mlir/IR/Builders.h"
#include "MyTf.h"
#include "ResourceType.h"
#include "VariantType.h"
#include "ReadVariableOp.h"
#include "AssignVariableOp.h"
#include "VarHandleOp.h"

namespace mlir {
  namespace mytf {

    MyTfType::MyTfType(MLIRContext *context):
      Dialect(getDialectNamespace(),context,TypeID::get<MyTfType>()) {
      addTypes < ResourceType >();
      addInterfaces<MyTfTypeInlinerInterface>();
    }

    Type MyTfType::parseType(DialectAsmParser &parser) const {
      Type t;
      if (succeeded(parser.parseOptionalKeyword("resource"))
	  && succeeded(parser.parseLess())
	  && succeeded(parser.parseType(t))
	  && succeeded(parser.parseGreater())) {
	return ResourceType::get(parser.getBuilder().getContext(), t);
      }
      else if (succeeded(parser.parseOptionalKeyword("variant"))
	  && succeeded(parser.parseLess())
	  && succeeded(parser.parseType(t))
	  && succeeded(parser.parseGreater())) {
	return ResourceType::get(parser.getBuilder().getContext(), t);
      }
      return Type();
    }

    void MyTfType::printType(Type type, DialectAsmPrinter &print) const {
      if (type.isa<ResourceType>()) {
	ResourceType rt = type.cast<ResourceType>();
	print << "resource<" << rt.getType() << ">";
      }
      else if (type.isa<VariantType>()) {
	VariantType rt = type.cast<VariantType>();
	print << "variant<" << rt.getType() << ">";
      }
    }
    
    MyTf::MyTf(MLIRContext *context): Dialect(getDialectNamespace(),
					      context,
					      TypeID::get<MyTf>()) {
      addOperations < VarHandleOp >();
      addOperations < ReadVariableOp, AssignVariableOp >();
      addInterfaces<MyTfInlinerInterface>();
      allowUnknownOperations();
    }
  }
}
    
