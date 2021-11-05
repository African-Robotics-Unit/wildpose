/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#pragma once

#include <list>
#include "Image.h"

#include "timing.hpp"

#include "helper_bmp.hpp"
#include "helper_ppm.hpp"
#include "helper_raw.hpp"
#include "helper_dir.hpp"
#include "helper_tiff.hpp"

#include "checks.h"

std::string generateOutputFileName(const char *pattern, unsigned idx);

template<class T, class Allocator >
fastStatus_t fvLoadImage(
	std::string imageFileName,
	std::string outputFileName,
	ImageT<T, Allocator>   &inputImg,

	unsigned	rawHeight,
	unsigned	rawWidth,
	unsigned	rawBits,

	bool info
) {
	if (!fileExist(imageFileName.c_str())) {
		fprintf(stderr, "Can not open input file\n");
		return FAST_IO_ERROR;
	}

	inputImg.inputFileName = imageFileName;
	inputImg.outputFileName = outputFileName;

	hostTimer_t timer = NULL;
	if (info) {
		timer = hostTimerCreate();
		hostTimerStart(timer);
	}

	const bool isPPM = compareFileExtension(imageFileName.c_str(), ".pgm") || compareFileExtension(imageFileName.c_str(), ".pgm16")
		|| compareFileExtension(imageFileName.c_str(), ".ppm") || compareFileExtension(imageFileName.c_str(), ".ppm16");

	if (compareFileExtension(imageFileName.c_str(), ".bmp") || compareFileExtension(imageFileName.c_str(), ".dib")) {
		CHECK_FAST_OPEN_FILE(fvLoadBMP(inputImg.data, inputImg.h, inputImg.w, inputImg.wPitch, inputImg.surfaceFmt, imageFileName.c_str()));
	} else if (isPPM) {
		CHECK_FAST_OPEN_FILE(fvLoadPPM(imageFileName.c_str(), inputImg.data, inputImg.w, inputImg.wPitch, inputImg.h, inputImg.surfaceFmt));
	} else if (compareFileExtension(imageFileName.c_str(), ".tif") || compareFileExtension(imageFileName.c_str(), ".tiff")) {
		CHECK_FAST_OPEN_FILE(fvLoadTIFF(imageFileName.c_str(), inputImg));
	} else if (compareFileExtension(imageFileName.c_str(), ".raw")) {
		inputImg.w = rawWidth;
		inputImg.wPitch = (rawBits * rawWidth) / 8;
		inputImg.h = rawHeight;
		inputImg.bitsPerChannel = rawBits;
		inputImg.isRaw = true;
		inputImg.surfaceFmt = IdentifySurface(rawBits, 1);
		CHECK_FAST_OPEN_FILE(fvLoadRaw(imageFileName.c_str(), inputImg.data, inputImg.w, inputImg.h, 1, rawBits));
	} else {
		if (info) hostTimerDestroy(timer);
		fprintf(stderr, "Input file has inappropriate extension\n");
		return FAST_UNSUPPORTED_FORMAT;
	}

	if (info) {
		const double totalTime = hostTimerEnd(timer);
		printf("File read time: %.2f ms\n\n", totalTime * 1000.0);
		hostTimerDestroy(timer);
	}

	inputImg.bitsPerChannel = GetBitsPerChannelFromSurface(inputImg.surfaceFmt);
	return FAST_OK;
}

template<class Allocator >
fastStatus_t fvLoadImages(
	char *folderName,
	const char *outputPattern,
	std::list<Image<Allocator>> &inputImg,
	unsigned widthDefault,
	unsigned heightDefault,
	unsigned bitsDefault,
	bool info
) {
	std::list<std::string> files;
	hostTimer_t timer = NULL;
	double loadTime = 0.;

	getFileList(folderName, files);

	if (files.size() == 0) {
		fprintf(stderr, "No input files found\n");
		return FAST_IO_ERROR;
	}

	if (info) {
		timer = hostTimerCreate();
		hostTimerStart(timer);
	}

	unsigned idx = 0;
	for (auto i = files.begin(); i != files.end(); ++i) {
		Image<Allocator> img;
		CHECK_FAST(fvLoadImage(*i, generateOutputFileName(outputPattern, idx), img, heightDefault, widthDefault, bitsDefault, info));
		inputImg.push_back(img);
		idx++;
	}

	if (info) {
		loadTime = hostTimerEnd(timer);
		printf("PGM images read time = %.2f ms\n\n", loadTime * 1000.0);
		hostTimerDestroy(timer);
	}

	files.clear();
	return FAST_OK;
}

template<class Allocator >
fastStatus_t fvSaveImageToFile(
	char    *imageFileName,
	std::unique_ptr<unsigned char, Allocator> &h_Surface,
	fastSurfaceFormat_t surfaceFmt,
	unsigned            surfaceHeight,
	unsigned            surfaceWidth,
	unsigned            surfacePitch8,
	bool info
) {
	bool isGray = GetNumberOfChannelsFromSurface(surfaceFmt) == 1;
	bool isTwoBytePerPixel = GetBytesPerChannelFromSurface(surfaceFmt) == 2;

	bool isBmp = compareFileExtension(imageFileName, ".bmp");
	bool isPgm = compareFileExtension(imageFileName, ".pgm");
	bool isPpm = compareFileExtension(imageFileName, ".ppm");
	bool isTiff = compareFileExtension(imageFileName, ".tif") || compareFileExtension(imageFileName, ".tiff");

	hostTimer_t timer = NULL;
	if (info) {
		timer = hostTimerCreate();
		hostTimerStart(timer);
	}

	if (isBmp) {
		if (isTwoBytePerPixel) {
			fprintf(stderr, "BMP file is not support two byte per channel format.\n");
			return FAST_UNSUPPORTED_FORMAT;
		}
		CHECK_FAST_SAVE_FILE(fvStoreBMP(imageFileName, h_Surface, surfaceFmt, surfaceHeight, surfaceWidth, surfacePitch8));
	} else if ((isPgm || isPpm)) {
	if (isPpm && isGray) {
		fprintf(stderr, "Unappropriate output file format. Set to PGM\n");
		unsigned pos = static_cast<unsigned>(strlen(imageFileName));
		while (imageFileName[pos] != 'p') pos--;
		imageFileName[pos] = 'g';
	}

	if (isPgm && !isGray) {
		fprintf(stderr, "Unappropriate output file format. Set to PPM\n");
		unsigned pos = static_cast<unsigned>(strlen(imageFileName));
		while (imageFileName[pos] != 'g') pos--;
		imageFileName[pos] = 'p';
	}

		CHECK_FAST_SAVE_FILE(fvSavePPM(imageFileName, h_Surface, surfaceFmt, surfaceWidth, surfacePitch8, surfaceHeight));
	} else if (isTiff) {
		CHECK_FAST_SAVE_FILE(fvSaveTIFF(imageFileName, h_Surface, surfaceFmt, surfaceWidth, surfacePitch8, surfaceHeight));
	} else {
		fprintf(stderr, "Input file has inappropriate extension\n");
		return FAST_UNSUPPORTED_FORMAT;
	}
	if (info) {
		double totalTime = hostTimerEnd(timer);

		printf("File write time: %.2f ms\n\n", totalTime * 1000.0);
		hostTimerDestroy(timer);
	}
	return FAST_OK;
}

template<class Allocator >
fastStatus_t fvSaveImageToFile(
	char    *imageFileName,
	std::unique_ptr<unsigned char, Allocator> &h_Surface,
	fastSurfaceFormat_t surfaceFmt,
	unsigned            bitsPerChannel,
	unsigned            surfaceHeight,
	unsigned            surfaceWidth,
	unsigned            surfacePitch8,
	bool info
) {
	bool isGray = GetNumberOfChannelsFromSurface(surfaceFmt) == 1;
	bool isTwoBytePerPixel = GetBytesPerChannelFromSurface(surfaceFmt) == 2;

	bool isBmp = compareFileExtension(imageFileName, ".bmp");
	bool isPgm = compareFileExtension(imageFileName, ".pgm");
	bool isPpm = compareFileExtension(imageFileName, ".ppm");

	hostTimer_t timer = NULL;
	double totalTime = 0.;
	if (info) timer = hostTimerCreate();

	if (isPpm && isGray) {
		fprintf(stderr, "Unappropriate output file format. Set to PGM\n");
		int pos = static_cast<unsigned>(strlen(imageFileName));
		while (imageFileName[pos] != 'p') pos--;
		imageFileName[pos] = 'g';
	}

	if (isPgm && !isGray) {
		fprintf(stderr, "Unappropriate output file format. Set to PPM\n");
		int pos = static_cast<unsigned>(strlen(imageFileName));
		while (imageFileName[pos] != 'g') pos--;
		imageFileName[pos] = 'p';
	}

	if (isBmp) {
		if (info) hostTimerStart(timer);
		if (isTwoBytePerPixel) {
			fprintf(stderr, "BMP file is not support two byte per channel format.\n");
			return FAST_UNSUPPORTED_FORMAT;
		}
		CHECK_FAST_SAVE_FILE(fvStoreBMP(imageFileName, h_Surface, surfaceFmt, surfaceHeight, surfaceWidth, surfacePitch8));

		if (info) {
			totalTime = hostTimerEnd(timer);
		}
	} else if ((isPgm || isPpm)) {
		if (info) hostTimerStart(timer);

		unsigned char *p = (unsigned char *)h_Surface.get();
		int bitsPerChannel_ = GetBitsPerChannelFromSurface(surfaceFmt);
		int nChannels = GetNumberOfChannelsFromSurface(surfaceFmt);

		bitsPerChannel_ = (bitsPerChannel == 0 || bitsPerChannel == bitsPerChannel_) ? bitsPerChannel_ : bitsPerChannel;

		if (!SavePPM(imageFileName, p, surfaceWidth, surfacePitch8, surfaceHeight, bitsPerChannel_, nChannels))
			return FAST_IO_ERROR;

		if (info) totalTime = hostTimerEnd(timer);
	} else {
		fprintf(stderr, "Input file has inappropriate extension\n");
		return FAST_UNSUPPORTED_FORMAT;
	}
	if (info) {
		printf("File write time: %.2f ms\n\n", totalTime * 1000.0);
		hostTimerDestroy(timer);
	}
	return FAST_OK;
}

template<class Allocator>
fastStatus_t fvSaveImageToFile(
	char *fname,
	Image<Allocator> &img
) {
	return fvSaveImageToFile(fname, img.data, img.surfaceFmt, img.h, img.w, img.wPitch, false);
}

template<typename T>
void CreateListForThread(std::list<T> &src, std::list<T> &dst, int index);

bool IsGrayUncompressedFormat(const char* fname);
bool IsColorUncompressedFormat(const char* fname);
