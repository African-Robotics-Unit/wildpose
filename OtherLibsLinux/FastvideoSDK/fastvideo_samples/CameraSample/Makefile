include ../Makefile.common

LD_FLAGS += -lGL -lglut -lGLEW -lpthread

ifeq ($(STATIC_LIB),1)
LD_FLAGS += $(TOP)bin/libfastvideo_mjpeg.a
else
LD_FLAGS += -lfastvideo_mjpeg -lpthread
endif
LD_FLAGS += -L$(FFMPEG_LIB) -lavformat -lavcodec -lavutil -lswresample

SOURCES_CPP += $(TOP)common/helper_jpeg/helper_jpeg_load.cpp
SOURCES_CPP += $(TOP)common/helper_jpeg/helper_jpeg_store.cpp

SOURCES_CPP += $(TOP)core_samples/helper_image/helper_pfm.cpp
SOURCES_CPP += $(TOP)core_samples/helper_image/helper_lut.cpp

SOURCES_CPP += $(TOP)core_samples/options/DebayerOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/BaseColorCorrectionOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/JpegEncoderOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/GrayscaleCorrectionOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/FfmpegOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/CameraSampleOptions.cpp

OBJECTS_CPP := $(patsubst $(TOP),,$(SOURCES_CPP:%.cpp=%.o))

all: $(OBJECTS_CPP)
	gcc $^ -o $(TOP)bin/CameraSample $(LD_FLAGS)
	make clean

clean:
	rm -f *.o
	rm -f $(TOP)common/helper_jpeg/*.o
	rm -f $(TOP)core_samples/*.o
	rm -f $(TOP)core_samples/helper_image/*.o
	rm -f $(TOP)core_samples/options/*.o

$(OBJECTS_CPP): %.o: %.cpp
	$(GCC) -c $(CC_FLAGS) -o $@ $<