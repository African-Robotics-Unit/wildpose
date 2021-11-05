include ../Makefile.common

SOURCES_CPP += $(TOP)core_samples/options/ConverterOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/ImageConverterSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/CropOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/CropSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/MatrixGeneratorOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/MatrixGeneratorSampleOptions.cpp

SOURCES_CPP += $(TOP)core_samples/helper_image/helper_pfm.cpp

OBJECTS_CPP := $(patsubst $(TOP),,$(SOURCES_CPP:%.cpp=%.o))

all: $(OBJECTS_CPP)
	gcc $^ -o $(TOP)/bin/ImageConverterSample $(LD_FLAGS)
	make clean

clean:
	rm -f *.o
	rm -f $(TOP)core_samples/*.o
	rm -f $(TOP)core_samples/helper_image/*.o
	rm -f $(TOP)core_samples/options/*.o

$(OBJECTS_CPP): %.o: %.cpp
	$(GCC) -c $(CC_FLAGS) -o $@ $<