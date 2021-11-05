include ../Makefile.common

CC_FLAGS += -I$(TOP)common/helper_jpeg/

ifeq ($(STATIC_LIB),1)
LD_FLAGS := $(TOP)bin/libfastvideo_mjpeg.a $(LD_FLAGS)
else
LD_FLAGS += -lfastvideo_mjpeg
endif

LD_FLAGS += -L$(FFMPEG_LIB) -lavformat -lavcodec -lavutil -lswresample

SOURCES_CPP += $(TOP)common/helper_jpeg/helper_jpeg_load.cpp
SOURCES_CPP += $(TOP)common/helper_jpeg/helper_jpeg_store.cpp

SOURCES_CPP += $(TOP)core_samples/options/JpegEncoderOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/FfmpegOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/FfmpegSampleOptions.cpp

OBJECTS_CPP := $(patsubst $(TOP),,$(SOURCES_CPP:%.cpp=%.o))

all: $(OBJECTS_CPP)
	gcc $^ -o $(TOP)/bin/FfmpegSample $(LD_FLAGS)
	make clean

clean:
	rm -f *.o
	rm -f $(TOP)common/helper_jpeg/*.o
	rm -f $(TOP)core_samples/*.o
	rm -f $(TOP)core_samples/helper_image/*.o
	rm -f $(TOP)core_samples/options/*.o

$(OBJECTS_CPP): %.o: %.cpp
	$(GCC) -c $(CC_FLAGS) -o $@ $<