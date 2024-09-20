include tools.config
include ccode.config

all: mlirlus config

config: libs.config ccode.config depend.config

libs.config:
	ls $(LLVM)/lib | grep "\.a$$" | sed 's/lib/-l/g' | sed 's/\.a/ \\/g' | sed '1 i\LIBS= \\' > $@

depend.config: $(CFILES) $(HEADERS)
	[ -f $@ ] || touch $@
	makedepend -f $@ --Y  -- $(CFILES) $(HEADERS)

include libs.config
include depend.config

mlirlus : $(MLIRLUS_MAIN_OBJ) $(OBJFILES) $(MLIROPT)
	@echo "Linking..."
	@$(LD) $(LDFLAGS) -o $@ $(MLIRLUS_MAIN_OBJ) $(OBJFILES) $(LLIBS) -lz -ltinfo

%.o: %.cpp
	@echo "Compiling $<..."
	@$(CPP) $(CFLAGS) -c $< -o $@

clean:
	@rm -f $(OBJFILES) *~ main externc.so
	@rm -f *.bak
