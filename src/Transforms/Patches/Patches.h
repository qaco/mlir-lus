// lus/LusCond.h - Custom passes defnitions -*- C++ -*- //

#ifndef PATCHES_PASSES_H
#define PATCHES_PASSES_H

#include <memory>

namespace mlir {
  class Pass;
  std::unique_ptr<mlir::Pass> createMemrefCopyPass();
  std::unique_ptr<mlir::Pass> createMemrefClonePass();
  std::unique_ptr<mlir::Pass> createMemrefStackPass();
  std::unique_ptr<mlir::Pass> createLinalgCopyPass();
}

#endif
