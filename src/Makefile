include Makefile.config

DIAL_DIR = Dialects/
DIAL_LUS_DIR = $(DIAL_DIR)Lus/
DIAL_PSSAO_DIR = $(DIAL_DIR)Pssa/
DIAL_SYNC_DIR = $(DIAL_DIR)Sync/
DIAL_MYTF_DIR = $(DIAL_DIR)MyTf/

TRANS_DIR = Transforms/
TRANS_LUS_LUS_DIR = $(TRANS_DIR)LusToLus/
TRANS_LUS_PSSA_DIR = $(TRANS_DIR)LusToPssa/
TRANS_LUS_STD_DIR = $(TRANS_DIR)LusToStd/
TRANS_PSSA_PSSA_DIR = $(TRANS_DIR)PssaToPssa/
TRANS_SYNC_STD_DIR = $(TRANS_DIR)SyncToStd/
TRANS_LUS_SYNC_DIR = $(TRANS_DIR)LusToSync/
TRANS_PSSA_SYNC_DIR = $(TRANS_DIR)PssaToSync/
TRANS_PSSA_STD_DIR = $(TRANS_DIR)PssaToStd/
TRANS_UTILITIES_DIR = $(TRANS_DIR)Utilities/

PASSES_DIR = $(TRANS_DIR)Passes/

TOOLS_DIR = Tools/

H_DIAL_MYTF = \
	$(DIAL_MYTF_DIR)ResourceType.h \
	$(DIAL_MYTF_DIR)VariantType.h \
	$(DIAL_MYTF_DIR)MyTf.h \
	$(DIAL_MYTF_DIR)ReadVariableOp.h \
	$(DIAL_MYTF_DIR)AssignVariableOp.h \
	$(DIAL_MYTF_DIR)VarHandleOp.h \

CPP_DIAL_MYTF = \
	$(DIAL_MYTF_DIR)MyTf.cpp \
	$(DIAL_MYTF_DIR)ReadVariableOp.cpp \
	$(DIAL_MYTF_DIR)AssignVariableOp.cpp \
	$(DIAL_MYTF_DIR)VarHandleOp.cpp \

H_DIAL_LUS = \
	$(DIAL_LUS_DIR)Clock.h \
	$(DIAL_LUS_DIR)ClockAnalysis.h \
	$(DIAL_LUS_DIR)ClockTree.h \
	$(DIAL_LUS_DIR)FbyOp.h \
	$(DIAL_LUS_DIR)PreOp.h \
	$(DIAL_LUS_DIR)OnClockOp.h \
	$(DIAL_LUS_DIR)OnClockNodeOp.h \
	$(DIAL_LUS_DIR)Instance.h \
	$(DIAL_LUS_DIR)InstanceTest.h \
	$(DIAL_LUS_DIR)KPeriodic.h \
	$(DIAL_LUS_DIR)lus.h \
	$(DIAL_LUS_DIR)Node.h \
	$(DIAL_LUS_DIR)NodeTest.h \
	$(DIAL_LUS_DIR)NodeType.h \
	$(DIAL_LUS_DIR)TestCondition.h \
	$(DIAL_LUS_DIR)WhenOp.h \
	$(DIAL_LUS_DIR)MergeOp.h \
	$(DIAL_LUS_DIR)WhenType.h \
	$(DIAL_LUS_DIR)Yield.h \
	$(DIAL_LUS_DIR)YieldType.h \
	$(DIAL_LUS_DIR)SignalTypes.h \
	$(DIAL_LUS_DIR)ClockAnnotation.h \
	$(DIAL_LUS_DIR)InitOp.h \
	$(DIAL_LUS_DIR)OutputOp.h \
	$(DIAL_LUS_DIR)KPeriodicOp.h \
	$(DIAL_LUS_DIR)Clocking/ClassicClocker.h \
	$(DIAL_LUS_DIR)Clocking/ClassicClock.h \

CPP_DIAL_LUS = \
	$(DIAL_LUS_DIR)Clock.cpp \
	$(DIAL_LUS_DIR)ClockAnalysis.cpp \
	$(DIAL_LUS_DIR)ClockTree.cpp \
	$(DIAL_LUS_DIR)FbyOp.cpp \
	$(DIAL_LUS_DIR)PreOp.cpp \
	$(DIAL_LUS_DIR)OnClockOp.cpp \
	$(DIAL_LUS_DIR)OnClockNodeOp.cpp \
	$(DIAL_LUS_DIR)Instance.cpp \
	$(DIAL_LUS_DIR)InstanceTest.cpp \
	$(DIAL_LUS_DIR)KPeriodic.cpp \
	$(DIAL_LUS_DIR)lus.cpp \
	$(DIAL_LUS_DIR)Node.cpp \
	$(DIAL_LUS_DIR)NodeTest.cpp \
	$(DIAL_LUS_DIR)TestCondition.cpp \
	$(DIAL_LUS_DIR)ClockAnnotation.cpp \
	$(DIAL_LUS_DIR)WhenOp.cpp \
	$(DIAL_LUS_DIR)MergeOp.cpp \
	$(DIAL_LUS_DIR)Yield.cpp \
	$(DIAL_LUS_DIR)InitOp.cpp \
	$(DIAL_LUS_DIR)OutputOp.cpp \
	$(DIAL_LUS_DIR)KPeriodicOp.cpp \
	$(DIAL_LUS_DIR)Clocking/ClassicClocker.cpp \
	$(DIAL_LUS_DIR)Clocking/ClassicClock.cpp \

H_DIAL_PSSAO = \
	$(DIAL_PSSAO_DIR)pssa.h \
	$(DIAL_PSSAO_DIR)OutputOp.h \

CPP_DIAL_PSSAO = \
	$(DIAL_PSSAO_DIR)pssa.cpp \
	$(DIAL_PSSAO_DIR)OutputOp.cpp \

H_DIAL_SYNC = \
	$(DIAL_SYNC_DIR)SignalTypes.h \
	$(DIAL_SYNC_DIR)NodeType.h \
	$(DIAL_SYNC_DIR)Node.h \
	$(DIAL_SYNC_DIR)Sync.h \
	$(DIAL_SYNC_DIR)HaltOp.h \
	$(DIAL_SYNC_DIR)InputOp.h \
	$(DIAL_SYNC_DIR)OutputOp.h \
	$(DIAL_SYNC_DIR)TickOp.h \
	$(DIAL_SYNC_DIR)SyncOp.h \
	$(DIAL_SYNC_DIR)InOp.h \
	$(DIAL_SYNC_DIR)OutOp.h \
	$(DIAL_SYNC_DIR)InstOp.h \
	$(DIAL_SYNC_DIR)UndefOp.h \
	$(DIAL_SYNC_DIR)SelectOp.h \

CPP_DIAL_SYNC = \
	$(DIAL_SYNC_DIR)Sync.cpp \
	$(DIAL_SYNC_DIR)Node.cpp \
	$(DIAL_SYNC_DIR)InOp.cpp \
	$(DIAL_SYNC_DIR)InputOp.cpp \
	$(DIAL_SYNC_DIR)InstOp.cpp \
	$(DIAL_SYNC_DIR)OutOp.cpp \
	$(DIAL_SYNC_DIR)OutputOp.cpp \
	$(DIAL_SYNC_DIR)SyncOp.cpp \
	$(DIAL_SYNC_DIR)TickOp.cpp \
	$(DIAL_SYNC_DIR)SelectOp.cpp \
	$(DIAL_SYNC_DIR)HaltOp.cpp \
	$(DIAL_SYNC_DIR)UndefOp.cpp \

H_DIAL = $(H_DIAL_MYTF) $(H_DIAL_LUS) $(H_DIAL_PSSAO) $(H_DIAL_SYNC)
CPP_DIAL = $(CPP_DIAL_MYTF) $(CPP_DIAL_LUS) $(CPP_DIAL_PSSAO) $(CPP_DIAL_SYNC)

H_TOOLS = \
	$(TOOLS_DIR)ParserAux.h \
	$(TOOLS_DIR)CommandLine.h \

CPP_TOOLS = \
	$(TOOLS_DIR)ParserAux.cpp \
	$(TOOLS_DIR)CommandLine.cpp \

H_TRANS_LUS_LUS = \
	$(TRANS_LUS_LUS_DIR)SortAlongClocks.h \
	$(TRANS_LUS_LUS_DIR)InlineNodes.h \

CPP_TRANS_LUS_LUS = \
	$(TRANS_LUS_LUS_DIR)SortAlongClocks.cpp \
	$(TRANS_LUS_LUS_DIR)InlineNodes.cpp \

H_TRANS_SYNC_STD = \
	$(TRANS_SYNC_STD_DIR)NodeToFun.h \

CPP_TRANS_SYNC_STD = \
	$(TRANS_SYNC_STD_DIR)NodeToFun.cpp \

H_TRANS_UTILITIES = \
	$(TRANS_UTILITIES_DIR)Helpers.h \
	$(TRANS_UTILITIES_DIR)CondToPred.h \
	$(TRANS_UTILITIES_DIR)ConstantPool.h \
	$(TRANS_UTILITIES_DIR)ExtFunctionPool.h \
	$(TRANS_UTILITIES_DIR)BufferGenerator.h \

CPP_TRANS_UTILITIES = \
	$(TRANS_UTILITIES_DIR)Helpers.cpp \
	$(TRANS_UTILITIES_DIR)ConstantPool.cpp \
	$(TRANS_UTILITIES_DIR)ExtFunctionPool.cpp \
	$(TRANS_UTILITIES_DIR)BufferGenerator.cpp \

H_PASSES = \
	$(PASSES_DIR)Passes.h \

CPP_PASSES = \
	$(PASSES_DIR)CondactEquationsPass.cpp \
	$(PASSES_DIR)NormalizeSmarter.cpp \
	$(PASSES_DIR)NormalizeIOs.cpp \
	$(PASSES_DIR)ControlInversionPass.cpp \
	$(PASSES_DIR)SortAlongClocksPass.cpp \
	$(PASSES_DIR)InlineNodesPass.cpp \
	$(PASSES_DIR)GenEnvPass.cpp \
	$(PASSES_DIR)SyncToStdPass.cpp \
	$(PASSES_DIR)NodeToAutomatonPass.cpp \
	$(PASSES_DIR)GenIreeCPass.cpp \
	$(PASSES_DIR)ExpandMacrosPass.cpp \
	$(PASSES_DIR)ClassicClockCalculusPass.cpp \
	$(PASSES_DIR)AllFbysOnBaseClockPass.cpp \
	$(PASSES_DIR)CentralizedStatePass.cpp \
	$(PASSES_DIR)RecomputeOrderPass.cpp \
	$(PASSES_DIR)ExplicitSignalsPass.cpp \
	$(PASSES_DIR)ExplicitPredicatesPass.cpp \
	$(PASSES_DIR)SCFClocksPass.cpp \
	$(PASSES_DIR)NodeToReactiveFunctionPass.cpp \
	$(PASSES_DIR)InlineInstancesPass.cpp \
	$(PASSES_DIR)NodeToStepResetPass.cpp \

H_TRANS = \
	$(H_PASSES) \
	$(H_TRANS_LUS_LUS) \
	$(H_TRANS_UTILITIES) \
	$(H_TRANS_SYNC_STD) \
	$(H_TRANS_LUS_SYNC) \

CPP_TRANS = \
	$(CPP_PASSES) \
	$(CPP_TRANS_LUS_LUS) \
	$(CPP_TRANS_UTILITIES) \
	$(CPP_TRANS_SYNC_STD) \
	$(CPP_TRANS_LUS_SYNC) \

HEADERS = $(H_DIAL) $(H_TOOLS) $(H_TRANS)

CFILES = $(CPP_DIAL) $(CPP_TOOLS) $(CPP_TRANS)

OBJFILES = $(CFILES:.cpp=.o)

PATCHES_C_FILES = \
	$(TRANS_DIR)Patches/MemrefCopyPass.cpp \
	$(TRANS_DIR)Patches/MemrefClonePass.cpp \
	$(TRANS_DIR)Patches/MemrefStackPass.cpp \
	$(TRANS_DIR)Patches/LinalgCopyPass.cpp

PATCHES_OBJFILES = $(PATCHES_C_FILES:.cpp=.o)

all: mlirlus mlirprime #externc.so

depend: clean
	makedepend -f makefile.depend --Y  -- $(CFILES)

mlirlus : $(TOOLS_DIR)mlir-lus.o $(OBJFILES)
	@echo "Linking..."
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS) $(LIBS) $(LIBS)

mlirprime : $(TOOLS_DIR)mlir-prime.o $(PATCHES_OBJFILES)
	@echo "Linking..."
	@$(LD) $(LDFLAGS) -o $@ $^ $(LIBS) $(LIBS) $(LIBS) 

%.o: %.cpp
	@echo "Compiling $<..."
	@$(CPP) $(CFLAGS) -c $< -o $@

clean:
	@rm -f $(OBJFILES) *~ main externc.so
	@rm -f *.bak

include makefile.depend
# DO NOT DELETE
