include ../Makefile.common

SOURCES_CPP += $(TOP)core_samples/helper_image/helper_pfm.cpp

SOURCES_CPP += $(TOP)core_samples/options/DebayerOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/GrayscaleCorrectionOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/WhiteBalanceOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/DebayerSampleOptions.cpp

OBJECTS_CPP := $(patsubst $(TOP),,$(SOURCES_CPP:%.cpp=%.o))

all: $(OBJECTS_CPP)
	gcc $^ -o $(TOP)/bin/MuxSample $(LD_FLAGS)
	make clean

clean:
	rm -f *.o
	rm -f $(TOP)core_samples/*.o
	rm -f $(TOP)core_samples/helper_image/*.o
	rm -f $(TOP)core_samples/options/*.o

$(OBJECTS_CPP): %.o: %.cpp
	$(GCC) -c $(CC_FLAGS) -o $@ $<