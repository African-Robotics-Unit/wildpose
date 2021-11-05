include ../Makefile.common

ifeq ($(STATIC_LIB),1)
LD_FLAGS += $(TOP)bin/libfastvideo_denoise.a
else
LD_FLAGS += -lfastvideo_denoise
endif

SOURCES_CPP += $(TOP)core_samples/options/DenoiseOptions.cpp
SOURCES_CPP += $(TOP)common/EnumToStringDenoise.cpp
SOURCES_CPP += $(TOP)core_samples/StringToEnumDenoise.cpp

OBJECTS_CPP := $(patsubst $(TOP),,$(SOURCES_CPP:%.cpp=%.o))

all: $(OBJECTS_CPP)
	gcc $^ -o $(TOP)/bin/DenoiseSample $(LD_FLAGS)
	make clean

clean:
	rm -f *.o
	rm -f $(TOP)core_samples/*.o
	rm -f $(TOP)core_samples/helper_image/*.o
	rm -f $(TOP)core_samples/options/*.o

$(OBJECTS_CPP): %.o: %.cpp
	$(GCC) -c $(CC_FLAGS) -o $@ $<