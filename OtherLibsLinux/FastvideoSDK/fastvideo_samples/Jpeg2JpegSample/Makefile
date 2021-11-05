include ../Makefile.common

LD_FLAGS += -lpthread

CC_FLAGS += -I$(TOP)/common/helper_jpeg/

SOURCES_CPP += $(TOP)core_samples/barrier.cpp
SOURCES_CPP += $(TOP)common/helper_jpeg/helper_jpeg_load.cpp
SOURCES_CPP += $(TOP)common/helper_jpeg/helper_jpeg_store.cpp

SOURCES_CPP += $(TOP)core_samples/options/ImageFilterOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/ResizerOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/JpegEncoderOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/CropOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/Jpeg2JpegSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/PreloadJpeg.cpp

OBJECTS_CPP := $(patsubst $(TOP),,$(SOURCES_CPP:%.cpp=%.o))

all: $(OBJECTS_CPP)
	gcc $^ -o $(TOP)/bin/Jpeg2JpegSample $(LD_FLAGS)
	make clean

clean:
	rm -f *.o
	rm -f $(TOP)common/*.o
	rm -f $(TOP)common/helper_jpeg/*.o
	rm -f $(TOP)core_samples/*.o
	rm -f $(TOP)core_samples/helper_image/*.o
	rm -f $(TOP)core_samples/options/*.o

$(OBJECTS_CPP): %.o: %.cpp
	$(GCC) -c $(CC_FLAGS) -o $@ $<