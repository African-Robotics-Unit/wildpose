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

#include "Decoder.h"
#include "timing.hpp"
#include "checks.h"
#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "helper_jpeg.hpp"

inline unsigned Decoder::uDivUp(unsigned a, unsigned b) {
	return (a / b) + (a % b != 0);
}

fastStatus_t Decoder::Init(JpegDecoderSampleOptions &options, MtResult *result) {
	this->options = options;

	CHECK_FAST(fastJpegDecoderCreate(
		&hDecoder,

		options.SurfaceFmt == FAST_BGR8 ? FAST_RGB8 : options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,
		true,

		&dstBuffer
	));

	CHECK_FAST(fastExportToHostCreate(
		&hDeviceToHostAdapter,

		&surfaceFmt,

		dstBuffer
	));

	size_t requestedMemSpace = 0;
	CHECK_FAST(fastJpegDecoderGetAllocatedGpuMemorySize(hDecoder, &requestedMemSpace));
	const double megabyte = 1024.0 * 1024.0;
	if (mtMode && result != nullptr) {
		result->requestedMemSize = requestedMemSpace / megabyte;
	} else
		printf("Requested GPU memory space: %.2f MB\n\n", requestedMemSpace / megabyte);

	return FAST_OK;
}

fastStatus_t Decoder::Close() const {
	CHECK_FAST(fastJpegDecoderDestroy(hDecoder));
	CHECK_FAST(fastExportToHostDestroy(hDeviceToHostAdapter));

	return FAST_OK;
}

fastStatus_t Decoder::Decode(std::list< Bytestream<FastAllocator> > &inputImgs, std::list< Image<FastAllocator> > &outputImgs, int threadId, MtResult *result) {
	hostTimer_t host_timer = NULL;
	fastGpuTimerHandle_t deviceToHostTimer = NULL;
	fastGpuTimerHandle_t jpegDecoderTimer = NULL;

	float elapsedTimeLoadJpeg = 0.;
	float fullTime = 0.;

	float processAvgTimeGpu = 0.;
	float processMinTimeGpu = 99999.;
	float processMaxTimeGpu = -99999.;

	if (info) {
		host_timer = hostTimerCreate();
	}
	const hostTimer_t decode_timer = hostTimerCreate();
	fastGpuTimerCreate(&deviceToHostTimer);
	fastGpuTimerCreate(&jpegDecoderTimer);

	unsigned maxSize = 0;
	for (auto i = inputImgs.begin(); i != inputImgs.end(); ++i) {
		if ((*i).size > maxSize)
			maxSize = (*i).size;
	}

	FastAllocator alloc;
	unsigned char *h_Bytestream = static_cast<unsigned char *>(alloc.allocate(maxSize));

	Image<FastAllocator> img;
	img.data.reset(static_cast<unsigned char *>(alloc.allocate(options.MaxHeight * GetPitchFromSurface(surfaceFmt, options.MaxWidth) * sizeof(unsigned char))));

	double totalFileSize = 0;

	for (auto i = inputImgs.begin(); i != inputImgs.end(); ++i) {
		img.inputFileName = (*i).inputFileName;
		img.outputFileName = (*i).outputFileName;

		jfifInfo.bytestreamSize = (*i).size;
		jfifInfo.h_Bytestream = h_Bytestream;

		for (int images_count = 0; images_count < options.RepeatCount; images_count++) {
			if (info) {
				hostTimerStart(host_timer);
			}

			CHECK_FAST(fastJfifLoadFromMemory(
				(*i).data.get(),
				(*i).size,

				&jfifInfo
			));

			if (info) {
				elapsedTimeLoadJpeg = (float)hostTimerEnd(host_timer)*1000.0f + i->loadTimeMs;
			}

			if (threadId == 0 && images_count == 0) {
				printf("Input image: %s\nInput image size: %dx%d pixels\n", img.inputFileName.c_str(), jfifInfo.width, jfifInfo.height);
				printf("Input sampling format: %s\n", EnumToString(jfifInfo.jpegFmt));
				printf("Input restart interval: %d\n\n", jfifInfo.restartInterval);
			}

			if (jfifInfo.width > options.MaxWidth ||
				jfifInfo.height > options.MaxHeight) {
				fprintf(stderr, "Unsupported image size\n");
				continue;
			}

			img.w = jfifInfo.width;
			img.h = jfifInfo.height;
			img.surfaceFmt = surfaceFmt;

			img.wPitch = GetPitchFromSurface(surfaceFmt, img.w);

			hostTimerStart(decode_timer);
			if (!mtMode) {
				fastGpuTimerStart(jpegDecoderTimer);
			}

			CHECK_FAST(fastJpegDecode(
				hDecoder,

				&jfifInfo
			));

			if (!mtMode) {
				fastGpuTimerStop(jpegDecoderTimer);
				fastGpuTimerStart(deviceToHostTimer);
			}

			fastExportParameters_t exportParameters = { };
			exportParameters.convert = options.ConvertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
			CHECK_FAST(fastExportToHostCopy(
				hDeviceToHostAdapter,

				img.data.get(),
				img.w,
				img.wPitch,
				img.h,

				&exportParameters
			));

			if (!mtMode) {
				float elapsedDecodeGpu = 0.;
				float elapsedDeviceToHost = 0.;

				fastGpuTimerStop(deviceToHostTimer);
				float elapsedTotalDecodeTime = (float)hostTimerEnd(decode_timer)*1000.0f;
				fastGpuTimerGetTime(jpegDecoderTimer, &elapsedDecodeGpu);
				fastGpuTimerGetTime(deviceToHostTimer, &elapsedDeviceToHost);

				elapsedTotalDecodeTime -= elapsedDecodeGpu + elapsedDeviceToHost;
				float elapsedDecode = elapsedDecodeGpu + ((elapsedTotalDecodeTime > 0.0f) ? elapsedTotalDecodeTime : 0.0f); // in case of marks inserting on CPU
				if (info) {
					printf("JFIF load time from HDD to CPU memory = %.2f ms\n", elapsedTimeLoadJpeg);
					printf("Decode time (includes host-to-device transfer) = %.2f ms\n", elapsedDecode);
					printf("Device-To-Host transfer = %.2f ms\n\n", elapsedDeviceToHost);

					processMaxTimeGpu = std::max(processMaxTimeGpu, elapsedDecode);
					processMinTimeGpu = std::min(processMinTimeGpu, elapsedDecode);
					processAvgTimeGpu += elapsedDecode;
					fullTime += elapsedDeviceToHost;
				}
				fullTime += elapsedDecode;
			} else {
				fullTime += (float)hostTimerEnd(decode_timer)*1000.0f;
			}
		}

		if (options.RepeatCount == 1) {
			printf("Output image: %s\n\n", img.outputFileName.c_str());
		}
		outputImgs.push_back(img);

		for (unsigned j = 0; j < jfifInfo.exifSectionsCount; j++) {
			free(jfifInfo.exifSections[j].exifData);
			jfifInfo.exifSections[j].exifData = NULL;
		}

		if (jfifInfo.exifSections != NULL) {
			free(jfifInfo.exifSections);
			jfifInfo.exifSections = NULL;
			jfifInfo.exifSectionsCount = 0;
		}
		const unsigned channelCount = (jfifInfo.jpegFmt == FAST_JPEG_Y) ? 1 : 3;

		const double inSize = jfifInfo.width * jfifInfo.height* channelCount;
		totalFileSize += inSize * options.RepeatCount;
	}

	CHECK_FAST_DEALLOCATION(fastFree(h_Bytestream));

	const unsigned imageCount = static_cast<unsigned>(inputImgs.size() * options.RepeatCount);
	if (mtMode) {
		result->totalTime = fullTime;
		result->totalFileSize = totalFileSize;
	} else if (info) {
		printf("Total time for %d images = %.2f ms (without HDD I/O)\n", imageCount, fullTime);
		printf("Process time excluding device-to-host transfer: average = %.3f ms (%.0f FPS), min = %.3f ms, max = %.3f ms\n",
			processAvgTimeGpu / imageCount, imageCount / (processAvgTimeGpu / 1000.0), processMinTimeGpu, processMaxTimeGpu);
	} else
		printf("Process time for %d images without HDD I/O and excluding device-to-host transfer = %.2f ms; %.0f FPS\n", imageCount, fullTime, imageCount / (fullTime / 1000.0));

	if (info) {
		hostTimerDestroy(host_timer);
	}
	fastGpuTimerDestroy(deviceToHostTimer);
	fastGpuTimerDestroy(jpegDecoderTimer);
	hostTimerDestroy(decode_timer);

	return FAST_OK;
}
