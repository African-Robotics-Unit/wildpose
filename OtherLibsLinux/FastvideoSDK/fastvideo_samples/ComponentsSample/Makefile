include ../Makefile.common

CC_FLAGS += -I$(TOP)core_samples/PugiXml/

SOURCES_CPP += $(TOP)core_samples/helper_image/helper_pfm.cpp
SOURCES_CPP += $(TOP)core_samples/helper_image/helper_lut.cpp
SOURCES_CPP += $(TOP)core_samples/helper_image/helper_lut_3d.cpp
SOURCES_CPP += $(TOP)core_samples/helper_image/cube.cpp
SOURCES_CPP += $(TOP)core_samples/PugiXml/pugixml.cpp

SOURCES_CPP += $(TOP)core_samples/options/AffineOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/DefringeOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/LutOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/ToneCurveOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/BaseColorCorrectionOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/BadPixelCorrectionOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/BinningOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/DebayerOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/GrayscaleCorrectionOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/WhiteBalanceOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/BayerBlackShiftOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/BitDepthConverterOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/SelectChannelOptions.cpp

SOURCES_CPP += $(TOP)core_samples/options/DefringeSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/LutSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/ToneCurveSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/BaseColorCorrectionSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/BadPixelCorrectionSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/BinningSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/AffineSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/DebayerSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/LutDebayerSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/BitDepthConverterSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/BayerBlackShiftSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/SelectChannelSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/CropSampleOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/CropOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/ImageFilterOptions.cpp
SOURCES_CPP += $(TOP)core_samples/options/ImageFilterSampleOptions.cpp

OBJECTS_CPP := $(patsubst $(TOP),,$(SOURCES_CPP:%.cpp=%.o))

all: $(OBJECTS_CPP)
	gcc $^ -o $(TOP)/bin/ComponentsSample $(LD_FLAGS)
	make clean

clean:
	rm -f *.o
	rm -f $(TOP)core_samples/*.o
	rm -f $(TOP)core_samples/helper_image/*.o
	rm -f $(TOP)core_samples/options/*.o

$(OBJECTS_CPP): %.o: %.cpp
	$(GCC) -c $(CC_FLAGS) -o $@ $<
