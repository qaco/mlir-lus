#include "pssa.h"
#include "OutputOp.h"

namespace mlir {
  namespace pssa {

    Pssa::Pssa(MLIRContext *context) :
	Dialect(getDialectNamespace(),context,TypeID::get<Pssa>()) {
      addOperations<OutputOp>() ;
	addInterfaces<PssaInlinerInterface>();
      }
  }
}
