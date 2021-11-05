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
#include <sstream>

#include <cuda_runtime.h>

#include "JpegEncoderAsync.h"
#include "checks.h"
#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "helper_jpeg.hpp"

fastStatus_t JpegEncoderAsync::Init(JpegEncoderSampleOptions &options) {
	Quality = options.JpegEncoder.Quality;
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;

	jfifInfoAsync.jpegMode = FAST_JPEG_SEQUENTIAL_DCT;
	jfifInfoAsync.restartInterval = options.JpegEncoder.RestartInterval;
	jfifInfoAsync.jpegFmt = options.JpegEncoder.SamplingFmt;

	folder = options.IsFolder;

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

	printf("Requested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	CHECK_FAST_ALLOCATION(fastMalloc((void **)&jfifInfo.h_Bytestream, GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth) * options.MaxHeight * sizeof(unsigned char)));
	return FAST_OK;
}

fastStatus_t JpegEncoderAsync::Close(void) const {
	CHECK_FAST(fastJpegEncoderDestroy(hEncoder));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));

	if (jfifInfo.h_Bytestream != NULL) {
		CHECK_FAST_DEALLOCATION(fastFree(jfifInfo.h_Bytestream));
	}

	for (unsigned i = 0; i < jfifInfoAsync.exifSectionsCount; i++) {
		free(jfifInfoAsync.exifSections[i].exifData);
	}

	if (jfifInfoAsync.exifSections != NULL) {
		free(jfifInfoAsync.exifSections);
	}

	return FAST_OK;
}

fastStatus_t JpegEncoderAsync::Encode(std::list< Image<FastAllocator> > &inputImgs) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t jpegEncoderTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&importFromHostTimer);
		fastGpuTimerCreate(&jpegEncoderTimer);
	}

	for (auto i = inputImgs.begin(); i != inputImgs.end(); i++) {
		Image<FastAllocator> &img = *i;
		printf("Input image: %s\nInput image size: %dx%d pixels\n", img.inputFileName.c_str(), img.w, img.h);
		printf("Input sampling format: %s\n\n", EnumToString(img.samplingFmt));

		if (img.w > maxWidth ||
			img.h > maxHeight) {
			fprintf(stderr, "Unsupported image size\n");
			continue;
		}

		jfifInfoAsync.width = img.w;
		jfifInfoAsync.height = img.h;

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
			printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);

			fastGpuTimerStart(jpegEncoderTimer);
		}

		jfifInfoAsync.d_Bytestream = NULL; // there will be pointer to the GPU buffer
		CHECK_FAST(fastJpegEncodeAsync(
			hEncoder,

			Quality,
			&jfifInfoAsync
		));

		if (info) {
			fastGpuTimerStop(jpegEncoderTimer);
			fastGpuTimerGetTime(jpegEncoderTimer, &elapsedTimeGpu);

			const unsigned surfaceSize = img.h * img.w * GetNumberOfChannelsFromSurface(img.surfaceFmt);

			fullTime += elapsedTimeGpu;
			printf("Effective encoding performance (includes device-to-host transfer) = %.2f GB/s (%.2f ms)\n\n", double(surfaceSize) / elapsedTimeGpu * 1E-6, elapsedTimeGpu);
		}

		{
			// copy data from fastJfifInfoAsync_t to fastJfifInfo_t
			jfifInfo.jpegFmt = jfifInfoAsync.jpegFmt;

			jfifInfo.bytestreamSize = jfifInfoAsync.bytestreamSize;
			cudaError_t error = cudaMemcpy(jfifInfo.h_Bytestream, jfifInfoAsync.d_Bytestream, jfifInfo.bytestreamSize, cudaMemcpyDeviceToHost);
			if (error != cudaSuccess) {
				fprintf(stderr, "Can not copy data from device to host memory\n");
				return FAST_IO_ERROR;
			}

			jfifInfo.height = jfifInfoAsync.height;
			jfifInfo.width = jfifInfoAsync.width;
			jfifInfo.bitsPerChannel = jfifInfoAsync.bitsPerChannel;

			jfifInfo.exifSections = jfifInfoAsync.exifSections;
			jfifInfo.exifSectionsCount = jfifInfoAsync.exifSectionsCount;

			jfifInfo.quantState = jfifInfoAsync.quantState;
			jfifInfo.huffmanState = jfifInfoAsync.huffmanState;
			jfifInfo.scanMap = jfifInfoAsync.scanMap;
			jfifInfo.restartInterval = jfifInfoAsync.restartInterval;
		}

		printf("Output image: %s\n\n", img.outputFileName.c_str());
		CHECK_FAST_SAVE_FILE(fastJfifStoreToFile(
			img.outputFileName.c_str(),
			&jfifInfo
		));

		int inSize = fileSize(img.inputFileName.c_str());
		int outSize = fileSize(img.outputFileName.c_str());
		printf("Input file size: %.2f KB\nOutput file size: %.2f KB\nCompression ratio: %.2f\n\n", inSize / 1024.0, outSize / 1024.0, float(inSize) / outSize);
	}

	if (info) {
		printf("Total time for all images = %.2f ms\n", fullTime);
		fastGpuTimerDestroy(importFromHostTimer);
		fastGpuTimerDestroy(jpegEncoderTimer);
	}

	return FAST_OK;
}
