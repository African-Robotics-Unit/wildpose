include ../Makefile.common

LD_FLAGS += -lpthread -fopenmp

ifeq ($(STATIC_LIB),1)
LD_FLAGS += $(TOP)bin/libfastvideo_decoder_j2k.a
else
LD_FLAGS += -lfastvideo_decoder_j2k
endif

SOURCES_CPP += $(TOP)core_samples/barrier.cpp
SOURCES_CPP += $(TOP)core_samples/options/J2kDecoderOptions.cpp
SOURCES_CPP += $(TOP)core_samples/J2kDecoderHelper.cpp
SOURCES_CPP += $(TOP)core_samples/J2kPrintBoxes.cpp
SOURCES_CPP += $(TOP)core_samples/PreloadJ2k.cpp

OBJECTS_CPP := $(patsubst $(TOP),,$(SOURCES_CPP:%.cpp=%.o))

all: $(OBJECTS_CPP)
	gcc $^ -o $(TOP)/bin/J2KDecoderSample $(LD_FLAGS)
	make clean

clean:
	rm -f *.o
	rm -f $(TOP)core_samples/*.o
	rm -f $(TOP)core_samples/helper_image/*.o
	rm -f $(TOP)core_samples/options/*.o

$(OBJECTS_CPP): %.o: %.cpp
	$(GCC) -c $(CC_FLAGS) -o $@ $<