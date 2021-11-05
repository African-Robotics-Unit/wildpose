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

#include "BGRXImport.h"

#include "supported_files.hpp"
#include "checks.h"

fastStatus_t BGRXImport::Init(BaseOptions &options) {
	folder = options.IsFolder;
	convertToBGR = options.ConvertToBGR;
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;

	CHECK_FAST(fastImportFromHostCreate(
		&hImport,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	CHECK_FAST(fastExportToHostCreate(
		&hDeviceToHostAdapter,
		&surfaceFmt,
		srcBuffer
	));

	FastAllocator allocator;
	const unsigned pitch = GetPitchFromSurface(surfaceFmt, options.MaxWidth);
	CHECK_FAST_ALLOCATION(h_Result.reset(static_cast<unsigned char *>(allocator.allocate(pitch * options.MaxHeight * sizeof(unsigned char)))));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImport, &tmp));
	requestedMemSpace += tmp;

	CHECK_FAST(fastExportToHostGetAllocatedGpuMemorySize(hDeviceToHostAdapter, &tmp));
	requestedMemSpace += tmp;

	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t BGRXImport::Transform(Image<FastAllocator> &image) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t importTimer = NULL;
	fastGpuTimerHandle_t deviceToHostTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&importTimer);
		fastGpuTimerCreate(&deviceToHostTimer);
	}

	printf("Input image: %s\nImage size: %dx%d pixels\n\n", image.inputFileName.c_str(), image.w, image.h);

	if (info) {
		fastGpuTimerStart(importTimer);
	}

	CHECK_FAST(fastImportFromHostCopy(
		hImport,
		image.data.get(),

		image.w,
		GetPitchFromSurface(image.surfaceFmt, image.w),
		image.h
	));

	if (info) {
		fastGpuTimerStop(importTimer);
		fastGpuTimerGetTime(importTimer, &elapsedTimeGpu);

		fullTime += elapsedTimeGpu;
		printf("Raw unpacker time = %.2f ms\n", elapsedTimeGpu);

		fastGpuTimerStart(deviceToHostTimer);
	}

	fastExportParameters_t exportParameters = { };
	exportParameters.convert = convertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
	CHECK_FAST(fastExportToHostCopy(
		hDeviceToHostAdapter,

		h_Result.get(),
		image.w,
		GetPitchFromSurface(surfaceFmt, image.w),
		image.h,

		&exportParameters
	));

	if (info) {
		fastGpuTimerStop(deviceToHostTimer);
		fastGpuTimerGetTime(deviceToHostTimer, &elapsedTimeGpu);

		fullTime += elapsedTimeGpu;
		printf("Device-to-host transfer = %.2f ms\n\n", elapsedTimeGpu);
	}

	printf("Output image: %s\n\n", image.outputFileName.c_str());

	CHECK_FAST_SAVE_FILE(fvSaveImageToFile(
		const_cast<char *>(image.outputFileName.c_str()),
		h_Result,
		surfaceFmt,
		image.h,
		image.w,
		GetPitchFromSurface(surfaceFmt, image.w),
		false
	));

	if (info) {
		printf("Total time for all images = %.2f ms\n", fullTime);
		fastGpuTimerDestroy(importTimer);
		fastGpuTimerDestroy(deviceToHostTimer);
	}

	return FAST_OK;
}

fastStatus_t BGRXImport::Close(void) const {
	CHECK_FAST(fastImportFromHostDestroy(hImport));
	CHECK_FAST(fastExportToHostDestroy(hDeviceToHostAdapter));

	return FAST_OK;
}
