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
#include <math.h>

#include "FfmpegEncoder.h"
#include "timing.hpp"
#include "checks.h"
#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "helper_jpeg.hpp"

fastStatus_t FfmpegEncoder::Init(FfmpegSampleOptions &options) {
	Quality = options.JpegEncoder.Quality;
	jfifInfo.restartInterval = options.JpegEncoder.RestartInterval;
	jfifInfo.jpegFmt = options.JpegEncoder.GrayAsRGB? fastJpegFormat_t::FAST_JPEG_420:options.JpegEncoder.SamplingFmt;
	jfifInfo.jpegMode = FAST_JPEG_SEQUENTIAL_DCT;

	frameRepeat = options.Ffmpeg.FrameRepeat;
	frameRate = static_cast<int>(options.Ffmpeg.FrameRate);

	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;

	CHECK_FAST(fastImportFromHostCreate(
		&hHostToDeviceAdapter,

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

	fastMJpegFileDescriptor_t fileDescriptor = { 0 };
	fileDescriptor.fileName = options.OutputPath;
	fileDescriptor.height = options.MaxHeight;
	fileDescriptor.width = options.MaxWidth;
	fileDescriptor.samplingFmt = options.JpegEncoder.GrayAsRGB ? fastJpegFormat_t::FAST_JPEG_420 : options.JpegEncoder.SamplingFmt;

	CHECK_FAST(fastMJpegWriterCreate(
		&hMjpeg,

		&fileDescriptor,
		frameRate
	));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;

	CHECK_FAST(fastJpegEncoderGetAllocatedGpuMemorySize(hEncoder, &tmp));
	requestedMemSpace += tmp;

	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hHostToDeviceAdapter, &tmp));
	requestedMemSpace += tmp;

	printf("Requested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	const unsigned pitch = GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth);
	CHECK_FAST_ALLOCATION(fastMalloc((void **)&jfifInfo.h_Bytestream, pitch * options.MaxHeight * sizeof(unsigned char)));

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(h_Jpegstream.reset((unsigned char *)alloc.allocate(pitch * options.MaxHeight + JPEG_HEADER_SIZE)));
	JpegstreamSize = pitch * options.MaxHeight + JPEG_HEADER_SIZE;

	return FAST_OK;
}

fastStatus_t FfmpegEncoder::Close() const {
	CHECK_FAST(fastJpegEncoderDestroy(hEncoder));
	CHECK_FAST(fastImportFromHostDestroy(hHostToDeviceAdapter));

	CHECK_FAST(fastMJpegWriterClose(hMjpeg));

	if (jfifInfo.h_Bytestream != NULL)
		CHECK_FAST_DEALLOCATION(fastFree(jfifInfo.h_Bytestream));

	for (unsigned i = 0; i < jfifInfo.exifSectionsCount; i++) {
		free(jfifInfo.exifSections[i].exifData);
	}

	if (jfifInfo.exifSections != NULL) {
		free(jfifInfo.exifSections);
	}

	return FAST_OK;
}

fastStatus_t FfmpegEncoder::Encode(std::list< Image<FastAllocator> > &inputImg) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t hostToDeviceTimer = NULL;
	hostTimer_t timer = NULL;

	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
		timer = hostTimerCreate();
	}

	for (auto i = inputImg.begin(); i != inputImg.end(); ++i) {
		Image<FastAllocator> &img = *i;
		printf("Input image: %s\nInput image size: %dx%d pixels\n", img.inputFileName.c_str(), img.w, img.h);
		printf("Input sampling format: %s\n\n", EnumToString(img.samplingFmt));
		const unsigned channelCount = (img.samplingFmt == FAST_JPEG_Y) ? 1 : 3;

		if (img.w > maxWidth ||
			img.h > maxHeight) {
			fprintf(stderr, "Unsupported image size\n");
			continue;
		}

		for (int i = 0; i < frameRepeat; i++) {
			if (info) {
				fastGpuTimerStart(hostToDeviceTimer);
			}

			CHECK_FAST(fastImportFromHostCopy(
				hHostToDeviceAdapter,

				img.data.get(),
				img.w,
				img.wPitch,
				img.h
			));

			if (info) {
				fastGpuTimerStop(hostToDeviceTimer);
				fastGpuTimerGetTime(hostToDeviceTimer, &elapsedTimeGpu);

				fullTime += elapsedTimeGpu;
				printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);

				hostTimerStart(timer);
			}

			unsigned jpegSize = JpegstreamSize;
			jfifInfo.width = img.w;
			jfifInfo.height = img.h;

			CHECK_FAST(fastJpegEncode(
				hEncoder,

				Quality,
				&jfifInfo
			));

			if (info) {
				const float totalTime = (float)hostTimerEnd(timer);

				const unsigned surfaceSize = img.h * img.w * ((img.surfaceFmt == FAST_I8) ? 1 : 3);

				fullTime += totalTime * 1000.0f;
				printf("Effective encoding performance (includes device-to-host transfer) = %.2f GB/s (%.2f ms)\n\n", double(surfaceSize) / totalTime * 1E-9, totalTime * 1000.0);
			}

			CHECK_FAST(fastJfifStoreToMemory(
				h_Jpegstream.get(),
				&jpegSize,

				&jfifInfo
			));


			if (info) {
				hostTimerStart(timer);
			}

			CHECK_FAST(fastMJpegWriteFrame(
				hMjpeg,

				h_Jpegstream.get(),
				jpegSize
			));

			if (info) {
				const float totalTime = (float)hostTimerEnd(timer);
				printf("Write frame %d time = %.2f ms\n", i, totalTime * 1000.0);
			}
		}
	}

	if (info) {
		printf("Total time for all images = %.2f ms\n", fullTime);
		hostTimerDestroy(timer);
	}

	return FAST_OK;
}
