include ../Makefile.common

ifeq ($(STATIC_LIB),1)
LD_FLAGS += $(TOP)bin/libfastvideo_nppFilter.a
LD_FLAGS += $(TOP)bin/libfastvideo_nppResize.a
LD_FLAGS += $(TOP)bin/libfastvideo_nppRotate.a
LD_FLAGS += $(TOP)bin/libfastvideo_nppGeometry.a
LD_FLAGS += -lnppc
else
LD_FLAGS += -lfastvideo_nppFilter -lfastvideo_nppResize -lfastvideo_nppRotate -lfastvideo_nppGeometry
endif

SOURCES_CPP += $(wildcard ./../../core_samples/options/Npp*.cpp)
SOURCES_CPP += $(wildcard ./../../core_samples/options/InterpolationOptions.cpp)

OBJECTS_CPP := $(patsubst $(TOP),,$(SOURCES_CPP:%.cpp=%.o))

all: $(OBJECTS_CPP)
	$(GCC) $^ -o ./../../bin/NppSample -L./../../bin -L./../../lib/ $(LD_FLAGS) -lstdc++ -lm -lnppicc -lnppig -lnppif -Wl,-rpath,$(CUDA_PATH)/lib64
	make clean
clean:
	rm -f ./../core/*.o
	rm -f *.o

$(OBJECTS_CPP): %.o: %.cpp
	$(GCC) -c $(CC_FLAGS) -o $@ $<
