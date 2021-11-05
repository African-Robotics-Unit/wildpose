/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "supported_files.hpp"
#include "helper_ppm.hpp"
#include "helper_raw.hpp"
#include "ImageConverterSampleOptions.h"

#include "checks.h"
#include "SurfaceTraits.hpp"
#include "BaseAllocator.h"
#include "IdentifySurface.hpp"

#include "FastAllocator.h"
#include "MallocAllocator.h"

///////////////////////////////////////////////////////////
/// PGM/PPM section
///////////////////////////////////////////////////////////
fastStatus_t convertor16to12(Image<MallocAllocator> &inputImg, char *outputFile, unsigned shift, unsigned bitsPerChannel) {
	Image<MallocAllocator> outputImg;
	outputImg.CopyInfoAndAllocate(inputImg);

	const unsigned channels = GetNumberOfChannelsFromSurface(inputImg.surfaceFmt);
	unsigned bytePerChannel = uDivUp(bitsPerChannel, 8u);
	
	unsigned short *src = reinterpret_cast<unsigned short *>(inputImg.data.get());
	unsigned short *dest = reinterpret_cast<unsigned short *>(outputImg.data.get());

	for (unsigned i = 0; i < inputImg.h; i++) {
		for (unsigned j = 0; j < inputImg.w * channels; j++) {
			dest[i * inputImg.w * channels + j] = (src[i * inputImg.w * channels + j] >> shift);
		}
	}

	CHECK_FAST_SAVE_FILE(fvSaveImageToFile(outputFile, outputImg));
	printf("Conversion complete\n");

	return FAST_OK;
}


fastStatus_t convertor12to16(Image<MallocAllocator> &inputImg, char *outputFile, unsigned shift, unsigned bitsPerChannel) {
	Image<MallocAllocator> outputImg;
	outputImg.CopyInfoAndAllocate(inputImg);

	const unsigned channels = GetNumberOfChannelsFromSurface(inputImg.surfaceFmt);
	unsigned bytePerChannel = uDivUp(bitsPerChannel, 8u);

	unsigned short *src = reinterpret_cast<unsigned short *>(inputImg.data.get());
	unsigned short *dest = reinterpret_cast<unsigned short *>(outputImg.data.get());

	for (unsigned i = 0; i < inputImg.h; i++) {
		for (unsigned j = 0; j < inputImg.w * channels; j++) {
			dest[i * inputImg.w * channels + j] = (src[i * inputImg.w * channels + j] << shift);
		}
	}

	CHECK_FAST_SAVE_FILE(fvSaveImageToFile(outputFile, outputImg));
	printf("Conversion complete\n");

	return FAST_OK;
}

fastStatus_t convertor16to8(Image<MallocAllocator> &inputImg, char *outputFile, unsigned shift, unsigned bitsPerChannel) {
	Image<MallocAllocator> outputImg;

	outputImg.CopyInfo(inputImg);
	outputImg.bitsPerChannel = 8;
	const unsigned channels = GetNumberOfChannelsFromSurface(inputImg.surfaceFmt);
	outputImg.surfaceFmt = IdentifySurface(outputImg.bitsPerChannel, channels);
	outputImg.wPitch = GetPitchFromSurface(outputImg.surfaceFmt, outputImg.w);

	MallocAllocator alloc;
	CHECK_FAST_ALLOCATION(outputImg.data.reset((unsigned char *)alloc.allocate(outputImg.w * channels * outputImg.h)));

	unsigned char *src = inputImg.data.get();
	unsigned char *dest = outputImg.data.get();

	for (unsigned i = 0; i < inputImg.h; i++) {
		for (unsigned j = 0; j < inputImg.w * channels; j++) {
			const unsigned val = (src[i * inputImg.wPitch + j * 2] + (src[i * inputImg.wPitch + j * 2 + 1] << 8)) >> shift;
			dest[i * inputImg.w * channels + j] = val>0xFF ? 0xFF : val;
		}
	}

	CHECK_FAST_SAVE_FILE(fvSaveImageToFile(outputFile, outputImg));
	printf("Conversion complete\n");

	return FAST_OK;
}

fastStatus_t convertor8to16(Image<MallocAllocator> &inputImg, char *outputFile, unsigned shift, unsigned bitsPerChannel, bool randomize) {
	Image<MallocAllocator> outputImg;
	outputImg.CopyInfo(inputImg);

	outputImg.bitsPerChannel = bitsPerChannel;
	const unsigned channels = GetNumberOfChannelsFromSurface(inputImg.surfaceFmt);
	outputImg.surfaceFmt = IdentifySurface(outputImg.bitsPerChannel, channels);
	outputImg.wPitch = GetPitchFromSurface(outputImg.surfaceFmt, outputImg.w);

	MallocAllocator alloc;
	CHECK_FAST_ALLOCATION(outputImg.data.reset((unsigned char*)alloc.allocate(inputImg.w * channels * inputImg.h * sizeof(unsigned short))));

	unsigned char  *src = inputImg.data.get();
	unsigned short *dest = (unsigned short*)outputImg.data.get();

	unsigned char random = 0;
	const unsigned char randomShift = (bitsPerChannel == 16) ? 2 :0;
	const unsigned dstPitchE = inputImg.w * channels;
	for (unsigned i = 0; i < inputImg.h; i++) {
		for (unsigned j = 0; j < inputImg.w *channels; j++) {
			if (randomize) {
				dest[i * dstPitchE + j] = (src[i * inputImg.wPitch + j] << shift) | ((random << randomShift) & ((1 << shift) - 1));
				random++;
			} else {
				dest[i * dstPitchE + j] = src[i * inputImg.wPitch + j] << shift;
			}
		}
	}

	CHECK_FAST_SAVE_FILE(fvSaveImageToFile(outputFile, outputImg));
	printf("Conversion complete\n");

	return FAST_OK;
}

fastStatus_t bypass(Image<MallocAllocator> &inputImg, char *outputFile) {
	Image<MallocAllocator> outputImg(inputImg);
	CHECK_FAST_SAVE_FILE(fvSaveImageToFile(outputFile, outputImg));
	printf("Conversion complete\n");
	return FAST_OK;
}

fastStatus_t convert(char *inputFile, char *outputFile, unsigned shift, unsigned bitsPerChannel, bool randomize) {
	Image<MallocAllocator> inputImg;

	printf("Input file: %s\n", inputFile);
	printf("Output file: %s\n", outputFile);
	printf("Shift option: %d\n", shift);

	fastStatus_t ret;
	CHECK_FAST(fvLoadImage(std::string(inputFile), std::string(""), inputImg, 0, 0, 0, false));
	if (inputImg.bitsPerChannel > 8 && bitsPerChannel == 8)
		ret = convertor16to8(inputImg, outputFile, shift, bitsPerChannel);
	else if (inputImg.bitsPerChannel == 16 && bitsPerChannel > 8)
		ret = convertor16to12(inputImg, outputFile, shift, bitsPerChannel);
	else if (inputImg.bitsPerChannel == 8 && bitsPerChannel > 8)
		ret = convertor8to16(inputImg, outputFile, shift, bitsPerChannel, randomize);
	else if (inputImg.bitsPerChannel == 12 && bitsPerChannel > 12)
		ret = convertor12to16(inputImg, outputFile, shift, bitsPerChannel);
	else
		ret = bypass(inputImg, outputFile);
	return ret;
}

///////////////////////////////////////////////////////////
/// RAW section
///////////////////////////////////////////////////////////
fastStatus_t convertorRaw12ToPgm(char *inputFile, char *outputFile, unsigned w, unsigned h, unsigned bitsPerChannel) {
	Image<FastAllocator> inputImg;

	printf("Input file: %s (%d x %d pixels)\n", inputFile, w, h);
	printf("Bits per pixel: %d\n", bitsPerChannel);
	printf("Output file: %s\n", outputFile);

	CHECK_FAST(fvLoadImage(std::string(inputFile), std::string(""), inputImg, h, w, bitsPerChannel, false));

	FastAllocator alloc;
	Image<FastAllocator> outputImg;
	outputImg.data.reset((unsigned char*)alloc.allocate(inputImg.h*inputImg.w * sizeof(unsigned short)));
	outputImg.w = inputImg.w;
	outputImg.h = inputImg.h;
	outputImg.wPitch = inputImg.w * sizeof(unsigned short);

	unsigned char *_rawData = reinterpret_cast<unsigned char *>(inputImg.data.get());
	unsigned short *_outputData = reinterpret_cast<unsigned short *>(outputImg.data.get());

	int packedPos = 0;
	for (unsigned i = 0; i < outputImg.h; i++)
		for (unsigned j = 0; j < outputImg.w; j += 2) {
			_outputData[i*outputImg.wPitch / sizeof(unsigned short) + j] = _rawData[packedPos] + ((_rawData[packedPos + 1] & 0x0F) << 8);
			_outputData[i*outputImg.wPitch / sizeof(unsigned short) + j + 1] = (_rawData[packedPos + 2] << 4) + ((_rawData[packedPos + 1] & 0xF0) >> 4);
			packedPos += 3;
		}

	CHECK_FAST_SAVE_FILE(fvSavePPM(outputFile, outputImg.data, IdentifySurface(bitsPerChannel, 1), outputImg.w, outputImg.wPitch, outputImg.h));
	printf("Conversion complete\n");

	return FAST_OK;
}

fastStatus_t convertorRaw16ToPgm(char *inputFile, char *outputFile, unsigned w, unsigned h, unsigned bitsPerChannel) {
	Image<FastAllocator> inputImg;

	printf("Input file: %s (%d x %d pixels)\n", inputFile, w, h);
	printf("Bits per pixel: %d\n", bitsPerChannel);
	printf("Output file: %s\n", outputFile);

	CHECK_FAST(fvLoadImage(std::string(inputFile), std::string(""), inputImg, h, w, bitsPerChannel, false));
	
	FastAllocator alloc;
	Image<FastAllocator> outputImg;
	outputImg.data.reset((unsigned char*)alloc.allocate(inputImg.h*inputImg.w * sizeof(unsigned short)));
	outputImg.w = inputImg.w;
	outputImg.h = inputImg.h;
	outputImg.wPitch = inputImg.w * sizeof(unsigned short);

	unsigned short *_rawData = reinterpret_cast<unsigned short *>(inputImg.data.get());
	unsigned short *_outputData = reinterpret_cast<unsigned short *>(outputImg.data.get());


	for (unsigned i = 0; i < outputImg.h; i++)
		for (unsigned j = 0; j < outputImg.w; j += 2) {
			_outputData[i*outputImg.wPitch / sizeof(unsigned short) + j] = _rawData[i*outputImg.wPitch / sizeof(unsigned short) + j];
			_outputData[i*outputImg.wPitch / sizeof(unsigned short) + j + 1] = _rawData[i*outputImg.wPitch / sizeof(unsigned short) + j + 1];
		}

	CHECK_FAST_SAVE_FILE(fvSavePPM(outputFile, outputImg.data, IdentifySurface(bitsPerChannel, 1), outputImg.w, outputImg.wPitch, outputImg.h));
	printf("Conversion complete\n");

	return FAST_OK;
}

fastStatus_t convertorPgmToRaw(const ImageConverterSampleOptions options) {
	printf("Input file: %s\n", options.InputPath);
	printf("Output file: %s\n", options.OutputPath);

	Image<FastAllocator> inputImg;
	CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(""), inputImg, 0, 0, 0, false));
	if (inputImg.w % 2 != 0) {
		fprintf(stderr, "Image should have even width\n");
		return FAST_INVALID_SIZE;
	}

	if (inputImg.bitsPerChannel != 12) {
		fprintf(stderr, "Only 12 bit pgm can be converted to RAW\n");
		return FAST_INVALID_FORMAT;
	}

	const unsigned sizeInBytes = uDivUp(inputImg.w * inputImg.bitsPerChannel * inputImg.h, 8u);

	FastAllocator alloc;
	std::unique_ptr<unsigned char, FastAllocator> rawData;
	rawData.reset(static_cast<unsigned char*>(alloc.allocate(sizeInBytes)));

	unsigned char* dst = static_cast<unsigned char*>(rawData.get());
	unsigned short* src = reinterpret_cast<unsigned short*>(inputImg.data.get());

	const unsigned srcPitch = inputImg.wPitch / sizeof(unsigned short);

	int packedPos = 0;
	for (unsigned i = 0; i < inputImg.h; i++)
		for (unsigned j = 0; j < inputImg.w; j += 2) {
			if (options.ImageConverter.RawFormat == FAST_RAW_XIMEA12) {
				dst[packedPos] = src[i * srcPitch + j] & 0xFF;
				dst[packedPos + 1] = (src[i * srcPitch + j] >> 8) + ((src[i * srcPitch + j + 1] << 4) & 0xFF);
				dst[packedPos + 2] = (src[i * srcPitch + j + 1] >> 4) & 0xFF;
			} else {
				dst[packedPos] = (src[i * srcPitch + j] >> 4) & 0xFF;
				dst[packedPos + 1] = (src[i * srcPitch + j + 1] & 0x0F) + ((src[i * srcPitch + j] & 0x0F) << 4);
				dst[packedPos + 2] = (src[i * srcPitch + j + 1] >> 4) & 0xFF;
			}
			packedPos += 3;
		}

	CHECK_FAST_SAVE_FILE(fvSaveRaw(options.OutputPath, dst, inputImg.w, inputImg.h, 1, inputImg.bitsPerChannel));
	printf("Conversion complete\n");

	return FAST_OK;
}

///////////////////////////////////////////////////////////
/// 
///////////////////////////////////////////////////////////
fastStatus_t RunConversion(ImageConverterSampleOptions options) {
	if (options.BitsPerChannel == 12 && compareFileExtension(options.InputPath, ".raw")) {
		return convertorRaw12ToPgm(options.InputPath, options.OutputPath, options.RawWidth, options.RawHeight, options.BitsPerChannel);
	}
	if (options.BitsPerChannel == 16 && compareFileExtension(options.InputPath, ".raw")) {
		return convertorRaw16ToPgm(options.InputPath, options.OutputPath, options.RawWidth, options.RawHeight, options.BitsPerChannel);
	}

	if (compareFileExtension(options.InputPath, ".pgm") && compareFileExtension(options.OutputPath, ".raw")) {
		return convertorPgmToRaw(options);
	}

	if ((IsGrayUncompressedFormat(options.InputPath) && IsGrayUncompressedFormat(options.OutputPath)) ||
		(IsColorUncompressedFormat(options.InputPath) && IsColorUncompressedFormat(options.OutputPath))) {
		return convert(options.InputPath, options.OutputPath, options.ImageConverter.Shift, options.BitsPerChannel, options.ImageConverter.Randomize);
	}

	fprintf(stderr, "Incorrect input parameters. Supported conversions: RAW <-> PGM16, PPM(PGM) <-> PPM16(PGM16)\n");
	return FAST_INVALID_VALUE;
}
 