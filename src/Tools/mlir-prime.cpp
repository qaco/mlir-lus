#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "../Transforms/Patches/Patches.h"
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
memrefCopy("remove-simple-memref-copy",
	     llvm::cl::desc("Remove simpler memref copy."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
memrefClone("remove-memref-clone",
	     llvm::cl::desc("Remove memref clones."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
toStack("heap-to-stack",
	     llvm::cl::desc("Heap allocations to stack allocations."),
	     llvm::cl::init(false));
static llvm::cl::opt<bool>
linalgCopy("remove-simple-linalg-copy",
	     llvm::cl::desc("Remove simpler linalg copy."),
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


  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();
  context.getOrLoadDialect<mlir::complex::ComplexDialect>();

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
    
    mlir::OpPassManager &funcPM = pm.nest<mlir::FuncOp>();
    
    if(memrefCopy) {
      funcPM.addPass(mlir::createMemrefCopyPass());
    }
    if(memrefClone) {
      funcPM.addPass(mlir::createMemrefClonePass());
    }
    if(linalgCopy) {
      funcPM.addPass(mlir::createLinalgCopyPass());
    }
    if(toStack) {
      funcPM.addPass(mlir::createMemrefStackPass());
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
