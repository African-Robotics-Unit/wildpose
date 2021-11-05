/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <cstdio>
#include <time.h>
#include <libexif/exif-content.h>

#include "Encoder.h"
#include "Decoder.h"
#include "checks.h"
#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "helper_jpeg.hpp"
#include "helper_exif.hpp"

fastStatus_t Encoder::InitExif() {
	unsigned char* exifBytestream;
	unsigned int   exifBytestreamLen;

	ExifData* exifData = exif_data_new();

	exif_data_unset_option(exifData, EXIF_DATA_OPTION_FOLLOW_SPECIFICATION);
	exif_data_set_option(exifData, EXIF_DATA_OPTION_DONT_CHANGE_MAKER_NOTE);
	exif_data_set_byte_order(exifData, EXIF_BYTE_ORDER_MOTOROLA);

	const char *make = "fastvideo encoder";
	ExifEntry *entry = fastExifCreateAsciiTag (exifData, EXIF_IFD_0, EXIF_TAG_MAKE, static_cast<unsigned>(strlen(make) + 1));
	strcpy((char *)entry->data, make);

	time_t timer;
	time(&timer);
	struct tm * time = localtime(&timer);
	char dataStr[20];
	sprintf(static_cast<char*>(dataStr), "%4d:%02d:%02d %02d:%02d:%02d", time->tm_year + 1900, time->tm_mon + 1, time->tm_mday, time->tm_hour, time->tm_min, time->tm_sec); /*YYYY:MM:DD HH:MM:SS*/
	entry = fastExifCreateAsciiTag(exifData, EXIF_IFD_0, EXIF_TAG_DATE_TIME, static_cast<unsigned>(strlen(dataStr) + 1));
	strcpy((char *)entry->data, dataStr);

	char buffer[30];
	strcpy(static_cast<char *>(buffer), "ASCII");
	buffer[7] = 0;
	strcpy(static_cast<char *>(buffer) + 8, make);

	entry = fastExifCreateAsciiTag(exifData, EXIF_IFD_EXIF, EXIF_TAG_USER_COMMENT, static_cast<unsigned>(strlen(make) + 1 + 8));
	memcpy((char *)entry->data, buffer, strlen(make) + 1 + 8);

	exif_data_save_data(exifData, &exifBytestream, &exifBytestreamLen);

	jfifInfo.exifSections = new fastJpegExifSection_t[1];
	jfifInfo.exifSections[0].exifCode = EXIF_SECTION_CODE;
	jfifInfo.exifSections[0].exifLength = exifBytestreamLen;
	jfifInfo.exifSections[0].exifData = new char[exifBytestreamLen];
	memcpy(jfifInfo.exifSections[0].exifData, exifBytestream, exifBytestreamLen);

	jfifInfo.exifSectionsCount = 1;
	
	if (exifBytestream)
		free(exifBytestream);

	exif_data_free(exifData);

	return FAST_OK;
}

fastStatus_t Encoder::Init(JpegEncoderSampleOptions &options, MtResult *result) {
	this->options = options;

	memset(&jfifInfo, 0, sizeof(jfifInfo));
	jfifInfo.restartInterval = options.JpegEncoder.RestartInterval;
	jfifInfo.jpegFmt = options.JpegEncoder.SamplingFmt;
	jfifInfo.bitsPerChannel = GetBitsPerChannelFromSurface(options.SurfaceFmt);

	if (options.JpegEncoder.noExif) {
		jfifInfo.exifSections = NULL;
		jfifInfo.exifSectionsCount = 0;
	} else {
		InitExif();
	}

	CHECK_FAST(fastImportFromHostCreate(
		&hImportFromHost,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));
	if (options.SurfaceFmt == FAST_BGR8) {
		options.SurfaceFmt = FAST_RGB8;
	}

	CHECK_FAST(fastJpegEncoderCreate(
		&hEncoder,

		options.MaxWidth,
		options.MaxHeight,

		srcBuffer
	));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;

	CHECK_FAST(fastJpegEncoderGetAllocatedGpuMemorySize(hEncoder, &tmp));
	requestedMemSpace += tmp;

	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;

	const double megabyte = 1024.0 * 1024.0;
	if (mtMode && result != nullptr) {
		result->requestedMemSize = requestedMemSpace / megabyte;
	} else {
		printf("Requested GPU memory space: %.2f MB\n\n", requestedMemSpace / megabyte);
	}

	const unsigned bufSize = static_cast<unsigned>(GetPitchFromSurface(options.SurfaceFmt, uSnapUp(options.MaxWidth, 16u)) * uSnapUp(options.MaxHeight, 16u) * 1.5f);
	CHECK_FAST_ALLOCATION(fastMalloc((void **)&jfifInfo.h_Bytestream, bufSize * sizeof(unsigned char)));
	return FAST_OK;
}

fastStatus_t Encoder::Close(void) const {
	CHECK_FAST(fastJpegEncoderDestroy(hEncoder));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));

	if (jfifInfo.h_Bytestream != NULL) {
		CHECK_FAST_DEALLOCATION(fastFree(jfifInfo.h_Bytestream));
	}

	for (unsigned i = 0; i < jfifInfo.exifSectionsCount; i++) {
		free(jfifInfo.exifSections[i].exifData);
	}

	if (jfifInfo.exifSections != NULL) {
		free(jfifInfo.exifSections);
	}

	return FAST_OK;
}

fastStatus_t Encoder::Encode(std::list< Image<FastAllocator> > &inputImgs, fastJpegQuantState_t *quantState, int threadId, MtResult *result) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	float processAvgTimeGpu = 0.;
	float processMinTimeGpu = 99999.;
	float processMaxTimeGpu = -99999.;

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t jpegEncoderTimer = NULL;
	const hostTimer_t encodeTimer = hostTimerCreate();

	if (info) {
		CHECK_FAST(fastGpuTimerCreate(&importFromHostTimer));
	}

	CHECK_FAST(fastGpuTimerCreate(&jpegEncoderTimer));

	double totalFileSize = 0;
	for (auto i = inputImgs.begin(); i != inputImgs.end(); ++i) {
		Image<FastAllocator> &img = *i;
		if (threadId == 0) {
			printf("Input image: %s\nImage size: %dx%d pixels, %d bits\n\n", img.inputFileName.c_str(), img.w, img.h, img.bitsPerChannel);
			printf("Input sampling format: %s\n\n", EnumToString(img.samplingFmt));
		}

		if (img.w > options.MaxWidth ||
			img.h > options.MaxHeight) {
			fprintf(stderr, "Unsupported image size\n");
			continue;
		}

		jfifInfo.width = img.w;
		jfifInfo.height = img.h;

		hostTimerStart(encodeTimer);
		for (int images_count = 0; images_count < options.RepeatCount; images_count++) {
			if (info) {
				fastGpuTimerStart(importFromHostTimer);
			}

			CHECK_FAST(fastImportFromHostCopy(
				hImportFromHost,

				img.data.get(),
				img.w,
				img.wPitch,
				img.h
			));

			if (info) {
				fastGpuTimerStop(importFromHostTimer);
				fastGpuTimerGetTime(importFromHostTimer, &elapsedTimeGpu);

				fullTime += elapsedTimeGpu;
				printf("Host-to-device transfer = %.2f ms\n", elapsedTimeGpu);
			}

			if (!mtMode) {
				fastGpuTimerStart(jpegEncoderTimer);
			}

			if (quantState != NULL) {
				CHECK_FAST(fastJpegEncodeWithQuantTable(
					hEncoder,
					quantState,
					&jfifInfo
				));
			} else {
				CHECK_FAST(fastJpegEncode(
					hEncoder,

					options.JpegEncoder.Quality,
					&jfifInfo
				));
			}

			if (!mtMode) {
				fastGpuTimerStop(jpegEncoderTimer);
				fastGpuTimerGetTime(jpegEncoderTimer, &elapsedTimeGpu);

				fullTime += elapsedTimeGpu;
			}

			if (info) {
				processMaxTimeGpu = std::max(processMaxTimeGpu, elapsedTimeGpu);
				processMinTimeGpu = std::min(processMinTimeGpu, elapsedTimeGpu);
				processAvgTimeGpu += elapsedTimeGpu;
				const unsigned surfaceSize = img.h * img.w * GetNumberOfChannelsFromSurface(img.surfaceFmt);
				printf("Effective encoding performance (includes device-to-host transfer) = %.2f GB/s (%.2f ms)\n\n", double(surfaceSize) / elapsedTimeGpu * 1E-6, elapsedTimeGpu);
			}
		}
		if (mtMode) {
			const float elapsedTime = (float)hostTimerEnd(encodeTimer)*1000.0f;
			fullTime += elapsedTime;
		}
		if (!options.Discard) {
			printf("Output image: %s\n\n", img.outputFileName.c_str());
			CHECK_FAST_SAVE_FILE(fastJfifStoreToFile(
				img.outputFileName.c_str(),
				&jfifInfo
			));
		}

		const double inSize = fileSize(img.inputFileName.c_str());
		const int outSize = fileSize(img.outputFileName.c_str());
		printf("Input file size: %.2f KB\nOutput file size: %.2f KB\nCompression ratio: %.2f\n\n", inSize / 1024.0, outSize / 1024.0, float(inSize) / outSize);

		totalFileSize += inSize * options.RepeatCount;
	}

	const unsigned imageCount = static_cast<unsigned>(inputImgs.size() * options.RepeatCount);
	const double totalFileSizeMb = totalFileSize / (1024.0*1024.0);
	if (mtMode) {
		result->totalTime = fullTime;
		result->totalFileSize = totalFileSize;
	} else if (info) {
		printf("Processing time on GPU for %d images including all transfers = %.2f ms; %.0f MB/s;  %.0f FPS \n",
			imageCount, fullTime, double(totalFileSizeMb) / (fullTime * 1E-3), imageCount / (fullTime / 1000.0));

		printf("Processing time on GPU for %d images excluding host-to-device transfer: average = %.3f ms (%.0f MB/s;  %.0f FPS), min = %.3f ms, max = %.3f ms\n",
			imageCount, processAvgTimeGpu / imageCount, double(totalFileSizeMb) / (processAvgTimeGpu * 1E-3), imageCount / (processAvgTimeGpu * 1E-3),
			processMinTimeGpu, processMaxTimeGpu);
	} else
		printf("Processing time on GPU for %d images excluding host-to-device transfer = %.2f ms; %.0f MB/s;  %.0f FPS\n",
			imageCount, fullTime,
			double(totalFileSizeMb) / (fullTime * 1E-3),
			imageCount / (fullTime / 1000.0));

	if (info) {
		CHECK_FAST(fastGpuTimerDestroy(importFromHostTimer));
	}
	CHECK_FAST(fastGpuTimerDestroy(jpegEncoderTimer));
	hostTimerDestroy(encodeTimer);

	return FAST_OK;
}
