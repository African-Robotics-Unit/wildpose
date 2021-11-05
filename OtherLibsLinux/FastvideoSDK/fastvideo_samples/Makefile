FULL_OBJECT = bayerCompressionSample.o cameraMultiSample.o cameraSample.o componentsSample.o
FULL_OBJECT += debayerJpegSample.o debayerSample.o denoiseSample.o ffmpegSample.o imageConverterSample.o
FULL_OBJECT += jpegAsyncSample.o jpegSample.o muxSample.o jpeg2JpegSample.o
FULL_OBJECT += rawImportSample.o resizeSample.o sdiConverterSample.o histogramSample.o j2kDecoderSample.o
FULL_OBJECT += nppSample.o j2kEncoderSample.o tiff-ln.o

HOST_ARCH := $(shell uname -m)

all: $(FULL_OBJECT)

clean:
	cd ./BayerCompressionSample/ && $(MAKE) clean
	cd ./CameraMultiSample/ && $(MAKE) clean
	cd ./CameraSample/ && $(MAKE) clean
	cd ./ComponentsSample/ && $(MAKE) clean
	cd ./DebayerJpegSample/ && $(MAKE) clean
	cd ./DebayerSample/ && $(MAKE) clean
	cd ./DenoiseSample/ && $(MAKE) clean
	cd ./FfmpegSample/ && $(MAKE) clean
	cd ./ImageConverterSample/ && $(MAKE) clean
	cd ./JpegAsyncSample/ && $(MAKE) clean
	cd ./JpegSample/ && $(MAKE) clean
	cd ./MuxSample/ && $(MAKE) clean
	cd ./Jpeg2JpegSample/ && $(MAKE) clean
	cd ./RawImportSample/ && $(MAKE) clean
	cd ./ResizeSample/ && $(MAKE) clean
	cd ./SDIConverterSample/ && $(MAKE) clean
	cd ./HistogramSample/ && $(MAKE) clean
	cd ./J2KEncoderSample/ && $(MAKE) clean
	cd ./J2KDecoderSample/ && $(MAKE) clean
	cd ./NppSample/ && $(MAKE) clean
	
tiff-ln.o:
	cd ../bin/ && ln -sf ../libs/libtiff/lib/linux/$(HOST_ARCH)/libtiff.so.5.7.0 libtiff.so.5
	cd ../bin/ && ln -sf libtiff.so.5 libtiff.so

denoiseSample.o:
	cd ./DenoiseSample/ && $(MAKE)
	
cameraMultiSample.o:
	cd ./CameraMultiSample/ && $(MAKE)
	
debayerJpegSample.o:
	cd ./DebayerJpegSample/ && $(MAKE)

debayerSample.o:
	cd ./DebayerSample/ && $(MAKE)

jpegSample.o:
	cd ./JpegSample/ && $(MAKE)

jpeg2JpegSample.o:
	cd ./Jpeg2JpegSample/ && $(MAKE)

ffmpegSample.o:
	cd ./FfmpegSample/ && $(MAKE)

cameraSample.o:
	cd ./CameraSample/ && $(MAKE)

componentsSample.o:
	cd ./ComponentsSample/ && $(MAKE)

bayerCompressionSample.o:
	cd ./BayerCompressionSample/ && $(MAKE)

imageConverterSample.o:
	cd ./ImageConverterSample/ && $(MAKE)
	
resizeSample.o:
	cd ./ResizeSample/ && $(MAKE)
	
sdiConverterSample.o:
	cd ./SDIConverterSample/ && $(MAKE)
	
muxSample.o:
	cd ./MuxSample/ && $(MAKE)
	
jpegAsyncSample.o:
	cd ./JpegAsyncSample/ && $(MAKE)
	
rawImportSample.o:
	cd ./RawImportSample/ && $(MAKE)

j2kDecoderSample.o:
	cd ./J2KDecoderSample/ && $(MAKE)

histogramSample.o:
	cd ./HistogramSample/ && $(MAKE)
	
j2kEncoderSample.o:
	cd ./J2KEncoderSample/ && $(MAKE)

nppSample.o:
	cd ./NppSample/ && $(MAKE)