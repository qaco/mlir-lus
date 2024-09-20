#include "lus.h"
#include "FbyOp.h"
#include "PreOp.h"
#include "WhenOp.h"
#include "MergeOp.h"
#include "InitOp.h"
#include "OnClockOp.h"
#include "OnClockNodeOp.h"
#include "OutputOp.h"
#include "Node.h"
#include "NodeTest.h"
#include "Instance.h"
#include "InstanceTest.h"
#include "KPeriodicOp.h"
#include "ClockType.h"

namespace mlir {
  namespace lus {

    Lus::Lus(MLIRContext *context) :
      Dialect(getDialectNamespace(),context,TypeID::get<Lus>()) {
      addTypes <NodeType,
		YieldType,
		WhenType,
		ClockType> () ;
      addOperations <
	PreOp,
	WhenOp,
	MergeOp,
	NodeOp,
	NodeTestOp,
	InitOp,
	YieldOp,
	FbyOp,
	OnClockOp,
	OnClockNodeOp,
	OutputOp,
	KperiodicOp,
	InstanceOp,
	InstanceTestOp>() ;
      addInterfaces<LusInlinerInterface>();
    }
    
    void Lus::printType(Type, DialectAsmPrinter &) const {

    }
  }
}
