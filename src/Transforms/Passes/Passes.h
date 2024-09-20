// lus/LusCond.h - Custom passes defnitions -*- C++ -*- //

#ifndef MLIRLUS_PASSES_H
#define MLIRLUS_PASSES_H

#include <memory>

namespace mlir {
  class Pass;
  namespace lus {
    std::unique_ptr<mlir::Pass> createClassicClockCalculusPass();
    std::unique_ptr<mlir::Pass> createAllFbysOnBaseClockPass();
    std::unique_ptr<mlir::Pass> createCentralizedStatePass();
    std::unique_ptr<mlir::Pass> createExplicitSignalsPass();
    std::unique_ptr<mlir::Pass> createExplicitPredicatesPass();
    std::unique_ptr<mlir::Pass> createSCFClocksPass();
    std::unique_ptr<mlir::Pass> createRecomputeOrderPass();
    std::unique_ptr<mlir::Pass> createNodeToReactiveFunctionPass();
    std::unique_ptr<mlir::Pass> createInlineInstancesPass();
    std::unique_ptr<mlir::Pass> createNodeToStepResetPass();
    
    std::unique_ptr<mlir::Pass> createGenIreeCPass();
    std::unique_ptr<mlir::Pass> createNormalizeSmarterPass();
    std::unique_ptr<mlir::Pass> createExpandMacrosPass();
    std::unique_ptr<mlir::Pass> createNormalizeIOsPass();
    std::unique_ptr<mlir::Pass> createControlInversionPass();
    
    std::unique_ptr<mlir::Pass> createNodeToAutomatonPass();
    std::unique_ptr<mlir::Pass> createSortAlongClocksPass();
    std::unique_ptr<mlir::Pass> createGenEnvPass();
    std::unique_ptr<mlir::Pass> createInlineNodesPass();
    std::unique_ptr<mlir::Pass> createNodeToFunPass();
    std::unique_ptr<mlir::Pass> createCondactEquationsPass();
  }
  namespace pssa {
    std::unique_ptr<mlir::Pass> createFusionCondactsPass();
  }
  namespace sync {
    // std::unique_ptr<mlir::Pass> createAutomatonToFuncsPass();
    std::unique_ptr<mlir::Pass> createSyncToStandardPass();
  }
}

#endif
