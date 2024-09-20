#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "../../Dialects/Lus/Node.h"

namespace mlir {
  namespace lus {

    struct GenIreeCPass : public PassWrapper< GenIreeCPass,
					      OperationPass<ModuleOp>> {
      void runOnOperation() override;
    };

    Operation *buildCCall(OpBuilder &b, Location loc,
			  TypeRange resultTy,
			  StringRef callee,
			  ValueRange operands) {
      OperationState state(loc, emitc::CallOp::getOperationName());
      emitc::CallOp::build(b, state,
			   resultTy, callee, {}, {}, operands);
      return b.createOperation(state);
    }

    Operation *buildCPointer(OpBuilder &b, Location loc,
			     Type resultTy,
			     Value operand) {
      StringAttr esper = b.getStringAttr("&");
      OperationState state(loc, emitc::ApplyOp::getOperationName());
      emitc::ApplyOp::build(b, state,
			    resultTy, esper,operand);
      return b.createOperation(state);
    }

    Operation *buildCConst(OpBuilder &b, Location loc,
			     Type resultTy,
			     StringRef lit) {
      emitc::OpaqueAttr litAttr = emitc::OpaqueAttr::get(b.getContext(),
							 lit);
      OperationState state(loc, emitc::ConstantOp::getOperationName());
      emitc::ConstantOp::build(b, state,
			       resultTy, litAttr);
      return b.createOperation(state);
    }

    void GenIreeCPass::runOnOperation() {

      // mlir::ConversionTarget target(getContext());
      // target.addLegalDialect<emitc::EmitCDialect>();
      
      auto mod = getOperation();

      mod.body().getBlocks().clear();
      mod.body().push_back(new Block);

      OpBuilder modb(mod.body());
      Location loc = modb.getUnknownLoc();

      Type sizeTy = emitc::OpaqueType::get(modb.getContext(), "size_t");
      Type argsTy = emitc::OpaqueType::get(modb.getContext(), "char*");
      Type stringView = emitc::OpaqueType::get(modb.getContext(),
					       "iree_string_view_t");
      // Type statusTy = emitc::OpaqueType::get(modb.getContext(),
      // 					     "iree_status_t");
      Type statusTy = modb.getI1Type();
      Type instOptTy = emitc::OpaqueType::get(modb.getContext(),
					      "iree_runtime_instance_options_t");
      Type instOptPtrTy = emitc::OpaqueType::get(modb.getContext(),
						 "iree_runtime_instance_options_t*");
      Type apiTy = emitc::OpaqueType::get(modb.getContext(),
					  "iree_api_version_t");
      Type runInstPtrTy = emitc::OpaqueType::get(modb.getContext(),
						 "iree_runtime_instance_t*");
      Type runInstPtrPtrTy = emitc::OpaqueType::get(modb.getContext(),
						    "iree_runtime_instance_t**");
      Type allocTy = emitc::OpaqueType::get(modb.getContext(),
					    "iree_allocator_t");
      Type devicePtrTy = emitc::OpaqueType::get(modb.getContext(),
						"iree_hal_device_t*");
      
      // Create includes
      modb.create<emitc::IncludeOp>(loc,"iree/runtime/api.h",true);

      // Create the main
      FunctionType mainType = modb.getFunctionType({argsTy, argsTy},
       						   {});
      OperationState mainState(loc, FuncOp::getOperationName());
      FuncOp::build(modb, mainState,
      		    StringAttr::get(modb.getContext(), "driver"),
      		    TypeAttr::get(mainType),
      		    modb.getStringAttr("private"));
      FuncOp main = dyn_cast<FuncOp>(modb.createOperation(mainState));
      main.addEntryBlock();
      OpBuilder mainb(main.body());

      // Fetch args
      Value arg1 = main.body().getArgument(0);
      Value arg2 = main.body().getArgument(1);
      buildCCall(mainb, loc, {stringView}, "iree_make_cstring_view", {arg1});
      buildCCall(mainb, loc, {stringView}, "iree_make_cstring_view", {arg2});

      Operation* statusOp = buildCCall(mainb, loc,
				       {statusTy},
				       "iree_ok_status",
				       {});
      
      // Instance configuration
      Operation* instOptOp = buildCCall(mainb, loc, {instOptTy},
					"get_instance_options", {});
      Operation* instOptPtrOp = buildCPointer(mainb, loc,
					      instOptPtrTy,
					      instOptOp->getResult(0));
      Operation* apiOp = buildCConst(mainb, loc, apiTy,
				     "IREE_API_VERSION_LATEST");
      buildCCall(mainb, loc,
		 {},
		 "iree_runtime_instance_options_initialize",
		 {apiOp->getResult(0), instOptPtrOp->getResult(0)});
      buildCCall(mainb, loc,
		 {},
		 "iree_runtime_instance_options_use_all_available_drivers",
		 {instOptPtrOp->getResult(0)});
      Operation* runInstPtrOp = buildCConst(mainb, loc, runInstPtrTy,
					    "NULL");
      scf::IfOp createInstOp = mainb.create<scf::IfOp>(loc,
						       statusTy,
						       statusOp->getResult(0),
						       true);
      {
	OpBuilder thenb(createInstOp.thenRegion());
	Operation* allocOp = buildCCall(thenb, loc, {allocTy},
					"iree_allocator_system", {});
	Operation* runInstPtrPtrOp = buildCPointer(thenb, loc,
						   runInstPtrPtrTy,
						   runInstPtrOp->getResult(0));
	Operation *statusOptmp = buildCCall(thenb, loc,
					    {statusTy},
					    "iree_runtime_instance_create",
					    {instOptPtrOp->getResult(0),
					     allocOp->getResult(0),
					     runInstPtrPtrOp->getResult(0)});
	thenb.create<scf::YieldOp>(loc,
				   statusOptmp->getResult(0));
      }
      {
      	OpBuilder elseb(createInstOp.elseRegion());
      	elseb.create<scf::YieldOp>(loc,
      				   statusOp->getResult(0));
      }
      statusOp = createInstOp.getOperation();
      Operation* devicePtrOp = buildCConst(mainb, loc, devicePtrTy,
					   "NULL");
      // scf::IfOp createDevOp = mainb.create<scf::IfOp>(loc,
      // 						      statusTy,
      // 						      statusOp->getResult(0),
      // 						      true);
      // {
      // 	OpBuilder thenb(createDevOp.thenRegion());
      // 	thenb.create<scf::YieldOp>(loc,
      // 				   statusOptmp->getResult(0));
      // }
      // {
      // 	OpBuilder elseb(createDevOp.elseRegion());
      // 	elseb.create<scf::YieldOp>(loc,
      // 				   statusOp->getResult(0));
      // }
      
      // Return of the main
      OperationState retState(loc, ReturnOp::getOperationName());
      ReturnOp::build(mainb, retState, {});
      mainb.createOperation(retState);
    }

    std::unique_ptr<Pass> createGenIreeCPass() {
      return std::make_unique<GenIreeCPass>();
    }
    
  }
}
