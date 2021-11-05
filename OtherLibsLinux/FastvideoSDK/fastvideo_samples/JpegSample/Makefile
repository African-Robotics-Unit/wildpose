include ../Makefile.common

CC_FLAGS += -I$(TOP)common/helper_jpeg/
CC_FLAGS += -I$(TOP)core_samples/helper_image/
CC_FLAGS += -I$(TOP)libs/libexif/inc/


LD_FLAGS += -lfastvideo_jpegCpuDecoder
LD_FLAGS += -L$(JPEG_CPU_LIB)/lib/linux/$(HOST_ARCH) -ljpeg12
LD_FLAGS += -lpthread

LD_FLAGS += -L$(TOP)/libs/libexif/lib/linux/$(HOST_ARCH)/ -lexif


SOURCES_CPP += $(TOP)core_samples/barrier.cpp
SOURCES_CPP += $(TOP)common/helper_jpeg/helper_jpeg_load.cpp
SOURCES_CPP += $(TOP)common/helper_jpeg/helper_jpeg_store.cpp


SOURCES_CPP += $(TOP)core_samples/helper_image/helper_quant_table.cpp
SOURCES_CPP += $(TOP)core_samples/helper_image/helper_exif.cpp
SOURCES_CPP += $(TOP)core_samples/helper_image/BayerImageResize.cpp


SOURCES_CPP += $(TOP)core_samples/options/JpegEncoderOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/JpegEncoderSampleOptions.cpp

SOURCES_CPP += $(TOP)core_samples/options/JpegDecoderOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/JpegDecoderSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/PreloadJpeg.cpp

OBJECTS_CPP := $(patsubst $(TOP),,$(SOURCES_CPP:%.cpp=%.o))

all: $(OBJECTS_CPP)
	gcc $^ -o $(TOP)/bin/JpegSample $(LD_FLAGS)
	make clean

clean:
	rm -f *.o
	rm -f $(TOP)common/helper_jpeg/*.o
	rm -f $(TOP)core_samples/*.o
	rm -f $(TOP)core_samples/helper_image/*.o
	rm -f $(TOP)core_samples/options/*.o

$(OBJECTS_CPP): %.o: %.cpp
	$(GCC) -c $(CC_FLAGS) -o $@ $<