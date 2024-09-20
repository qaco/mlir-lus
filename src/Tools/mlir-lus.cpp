#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "../Dialects/Lus/lus.h"
#include "../Dialects/Pssa/pssa.h"
#include "../Dialects/Sync/Sync.h"
#include "../Dialects/Sync/Node.h"
#include "../Dialects/Lus/Node.h"
#include "../Dialects/Lus/NodeTest.h"
#include "../Dialects/MyTf/MyTf.h"
#include "../Transforms/Passes/Passes.h"
#include "../Tools/CommandLine.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir ;
using namespace mlir::vector;

//=========================================================
// Accepted command line arguments:

//---------------------------------------------------------
// input and output files
static llvm::cl::opt<std::string>
inputFilename(llvm::cl::Positional,
	      llvm::cl::desc("First positional argument, it sets the name of the input file. By default, it is -, which means input is taken from standard input."),
	      llvm::cl::init("-"));
static llvm::cl::opt<std::string>
outputFilename("o",
	       llvm::cl::desc("Set output filename"),
	       llvm::cl::value_desc("filename"),
	       llvm::cl::init("-"));

//---------------------------------------------------------
// Processing pipeline control
static llvm::cl::opt<bool>
inlineNodes("inline-nodes",
	    llvm::cl::desc("Inlines all node instances that are not marked noinline."),
	    llvm::cl::init(false));
static llvm::cl::opt<bool>
inlineInstances("inline-instances",
		llvm::cl::desc("Inlines all node instances."),
		llvm::cl::init(false));
static llvm::cl::opt<bool>
ensureLusDom("normalize",
	     llvm::cl::desc("Normalize lus nodes of an MLIR file."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
classicClockCalculus("classic-clock-calculus",
	     llvm::cl::desc("The classic clock calculus inference."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
fbysOnBaseClock("all-fbys-on-base-clock",
	     llvm::cl::desc("Put all fbys on the base clock."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
fbysCentralization("fbys-centralization",
	     llvm::cl::desc("Centralize the representation of state."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
explicitSignals("explicit-signals",
	     llvm::cl::desc("Explicit the IO signals."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
explicitClocks("explicit-clocks",
	     llvm::cl::desc("Explicit the clocks."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
SCFClocks("scf-clocks",
	     llvm::cl::desc("Lower DF clocks to CF clocks."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
recomputeOrder("recompute-order",
	     llvm::cl::desc("Recompute the topological inside each node."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
nodeToSync("node-to-reactive-func",
	     llvm::cl::desc("Lower lus nodes to sync functions."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
nodeToStep("node-to-step-func",
	     llvm::cl::desc("Lower lus nodes to step functions."),
	     llvm::cl::init(false));

static llvm::cl::opt<bool>
condExec("conditional-execution",
	     llvm::cl::desc("Lower lus clocks to CF conditional execution."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
toAutomata("to-sync-automata",
	  llvm::cl::desc("Lowers lus operations (including nodes) to the sync dialect."),
	  llvm::cl::init(false));
static llvm::cl::opt<std::string>
futureMain("mainnode",
	   llvm::cl::desc("Upon non-modular lowering of lus nodes to std functions, sets the name of the node that will be lowered."),
	   llvm::cl::value_desc("A string or -"),
	   llvm::cl::init("-"));
static llvm::cl::opt<bool>
lowerSyncToStd("sync-to-std",
	  llvm::cl::desc("Lower sync operations and types to the standard dialect."),
	  llvm::cl::init(false));
static llvm::cl::opt<bool>
invertControl("invert-control",
	       llvm::cl::desc("Lower lus to std using an control-inversion approach."),
	       llvm::cl::init(false));
static llvm::cl::opt<bool>
genIreeC("gen-iree-c",
	       llvm::cl::desc("Generate IREE C driver."),
	       llvm::cl::init(false));

//---------------------------------------------------------
// Debugging options
static llvm::cl::opt<unsigned>
verbose("verbose",
	llvm::cl::desc("Print more tracing information in all stages. The integer argument is the verbosity level. Zero means no info."),
	llvm::cl::init(0));
static llvm::cl::opt<bool>
disableCA("disable-clock-analysis",
	  llvm::cl::desc("Disable clock analysis. As code generation depends on clock analysis, this can only be used in order to debug the parser."),
	  llvm::cl::init(false));
static llvm::cl::opt<bool>
showDialects("show-dialects",
	     llvm::cl::desc("Print the list of registered dialects and exit."),
	     llvm::cl::init(false));



//=========================================================
//
// This function may be called to register the MLIR passes with the
// global registry.  If you're building a compiler, you likely don't
// need this: you would build a pipeline programmatically without the
// need to register with the global registry, since it would already
// be calling the creation routine of the individual passes.  The
// global registry is interesting to interact with the command-line
// tools.
void registerAllPassesSpecialized() {
  // Init general passes
  createCanonicalizerPass();
  createCSEPass();
  createLoopUnrollPass();
  createLoopUnrollAndJamPass();
  createSimplifyAffineStructuresPass();
  createLoopFusionPass();
  createLoopInvariantCodeMotionPass();
  createAffineLoopInvariantCodeMotionPass();
  createPipelineDataTransferPass();
  createLowerAffinePass();
  createLoopTilingPass(0);
  createLoopCoalescingPass();
  createAffineDataCopyGenerationPass(0, 0);  
  createStripDebugInfoPass();
  createInlinerPass();

  createSuperVectorizePass({});
  createPrintOpStatsPass();
  createSymbolDCEPass();
  createLocationSnapshotPass({});
  
  // Linalg
  createLinalgTilingPass();
  createLinalgPromotionPass(false,false);
  createConvertLinalgToLoopsPass();
  createConvertLinalgToAffineLoopsPass();

  createConvertLinalgToParallelLoopsPass();
  createConvertLinalgToLLVMPass();

  // LoopOps
  createParallelLoopFusionPass();
  createParallelLoopSpecializationPass();
  createParallelLoopTilingPass();
}


//=========================================================
// Main driver
int main(int argc, char **argv) {
  
  //--------------------------------------------------
  // Parse command line. This must happen **after**
  // initialization of MLIR and LLVM, because init adds
  // options.
  mlir::registerPassManagerCLOptions();
  // The following line implements a command-line parser for
  // MLIR passes.
  PassPipelineCLParser passPipeline("", "Compiler passes to run");
  llvm::cl::ParseCommandLineOptions(argc,argv,
				    "Work-in-progress tool\n");

  verboseLevel = verbose ;
  if(disableCA) {
    // llvm::outs() << "Clock analysis disabled!\n";
    disableClockAnalysis = true ;
  }
  if (futureMain.getNumOccurrences() > 0) {
    hasMainNode = true;
    mainNode = futureMain.getValue();
  }


  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::lus::Lus>();
  context.getOrLoadDialect<mlir::pssa::Pssa>();
  context.getOrLoadDialect<mlir::sync::Sync>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();
    context.getOrLoadDialect<mlir::mytf::MyTf>();
  context.getOrLoadDialect<mlir::mytf::MyTfType>();

  context.allowUnregisteredDialects();
  
  //--------------------------------------------------
  // Set up input and output files - must be done after
  // parsing the command line.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> ifile =
    openInputFile(inputFilename, &errorMessage);
  if (!ifile ) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  } 
  std::unique_ptr<llvm::ToolOutputFile> ofile =
    openOutputFile(outputFilename, &errorMessage);
  if (!ofile ) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  
 
  //--------------------------------------------------
  // Load the input file into the context
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr,&context);
  sourceMgr.AddNewSourceBuffer(std::move(ifile), llvm::SMLoc());
  OwningModuleRef module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return -2;
  }
 
  //--------------------------------------------------
  //
  {
    // Apply any pass manager command line options.
    mlir::PassManager pm(&context);
    applyPassManagerCLOptions(pm);

    if(inlineNodes) {
      pm.addPass(mlir::lus::createInlineNodesPass());
    }
    
    mlir::OpPassManager &lusNodePM = pm.nest<mlir::lus::NodeOp>();
    mlir::OpPassManager &lusTestNodePM = pm.nest<mlir::lus::NodeTestOp>();

    if(inlineInstances) {
      lusTestNodePM.addPass(mlir::lus::createInlineInstancesPass());
    }
    
    if(classicClockCalculus) {
      lusTestNodePM.addPass(mlir::lus::createClassicClockCalculusPass());
    }

    if (fbysOnBaseClock) {
      lusTestNodePM.addPass(mlir::lus::createAllFbysOnBaseClockPass());
    }

    if (fbysCentralization) {
      lusTestNodePM.addPass(mlir::lus::createCentralizedStatePass());
      // if(!invertControl && !toAutomata && !nodeToSync)
      // 	lusTestNodePM.addPass(mlir::lus::createExpandMacrosPass());
    }

    if (explicitSignals) {
      lusTestNodePM.addPass(mlir::lus::createExplicitSignalsPass());
    }


    if (recomputeOrder) {
      lusTestNodePM.addPass(mlir::lus::createRecomputeOrderPass());
    }

    if (explicitClocks) {
      lusTestNodePM.addPass(mlir::lus::createExplicitPredicatesPass());
    }

    if (SCFClocks) {
      lusTestNodePM.addPass(mlir::lus::createSCFClocksPass());
    }

    if (nodeToSync) {
      pm.addPass(mlir::lus::createNodeToReactiveFunctionPass());
    }

    if (nodeToStep) {
      pm.addPass(mlir::lus::createNodeToStepResetPass());
    }
    
    if(ensureLusDom) {
      // lusNodePM.addPass(mlir::lus::createNormalizeIOsPass());
      lusNodePM.addPass(mlir::lus::createNormalizeSmarterPass());
      lusNodePM.addPass(mlir::lus::createSortAlongClocksPass());
    }
    if (condExec) {
      lusNodePM.addPass(mlir::lus::createCondactEquationsPass());
    }

    if (genIreeC) {
      pm.addPass(mlir::lus::createGenIreeCPass());
    }
    if (toAutomata) {
      lusNodePM.addPass(mlir::lus::createGenEnvPass());
    }
    if (toAutomata || invertControl) {
      lusNodePM.addPass(mlir::lus::createCondactEquationsPass());
      // pm.addPass(mlir::pssa::createFusionCondactsPass());
    }
    if (invertControl) {
      pm.addPass(mlir::lus::createControlInversionPass());
    }
    if (toAutomata) {
      pm.addPass(mlir::lus::createNodeToAutomatonPass());
    }

    if (lowerSyncToStd) {
      pm.addPass(mlir::sync::createSyncToStandardPass());
    }
    
    // Build the provided pipeline from standard command line
    // arguments
    function_ref<LogicalResult(const Twine &)> errorHandler;
    if (mlir::failed(passPipeline.addToPipeline(pm,errorHandler)))
      return -1 ;
    
    // Run the pipeline.
    if (mlir::failed(pm.run(*module)))
      return -2 ;
  }

  //--------------------------------------------------
  // Print the result in the output file. If no name is
  // provided, this file is the standard output. 
  // Note the use of the -> operator which is equivalent to
  // module.get().
  module->print(ofile->os()) ;
  ofile->os() << "\n" ;
  // Make sure the file is not deleted
  ofile->keep() ;

  // Normal return value
  return 0 ;
}
