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

#include "UnsharpMask.h"
#include "supported_files.hpp"
#include "checks.h"

#include "fastvideo_sdk.h"

fastStatus_t UnsharpMask::Init(NppImageFilterSampleOptions &options) {
	this->options = options;

	CHECK_FAST(fastGetSdkParametersHandle(&hSdkParameters));
	CHECK_FAST(fastImportFromHostCreate(
		&hImportFromHost,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	CHECK_FAST(fastNppFilterLibraryInit(hSdkParameters));
	CHECK_FAST(fastNppFilterCreate(
		&hUnsharpMask,
		options.NppImageFilter.Threshold == 0. ? NPP_UNSHARP_MASK_SOFT : NPP_UNSHARP_MASK_HARD,
		NULL,
		options.MaxWidth,
		options.MaxHeight,

		srcBuffer,
		&dstBuffer
	));

	CHECK_FAST(fastExportToHostCreate(
		&hExportToHost,

		&surfaceFmt,

		dstBuffer
	));

	const unsigned pitch = GetPitchFromSurface(surfaceFmt, options.MaxWidth);

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(h_Result.reset(static_cast<unsigned char *>(alloc.allocate(pitch * options.MaxHeight))));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;
	CHECK_FAST(fastNppFilterGetAllocatedGpuMemorySize(hUnsharpMask, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;
	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t UnsharpMask::Transform(
	std::list<Image<FastAllocator> > &image
) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t unsharpMaskTimer = NULL;
	fastGpuTimerHandle_t exportToHostTimer = NULL;

	if (info) {
		CHECK_FAST(fastGpuTimerCreate(&importFromHostTimer));
		CHECK_FAST(fastGpuTimerCreate(&unsharpMaskTimer));
		CHECK_FAST(fastGpuTimerCreate(&exportToHostTimer));
	}

	for (auto i = image.begin(); i != image.end(); ++i) {
		Image<FastAllocator> &img = *i;

		printf("Input image: %s\nImage size: %dx%d pixels\n\n", img.inputFileName.c_str(), img.w, img.h);

		if (img.w > options.MaxWidth ||
			img.h > options.MaxHeight) {
			fprintf(stderr, "Unsupported image size\n");
			continue;
		}

		if (info) {
			CHECK_FAST(fastGpuTimerStart(importFromHostTimer));
		}

		CHECK_FAST(fastImportFromHostCopy(
			hImportFromHost,

			img.data.get(),
			img.w,
			img.wPitch,
			img.h
		));

		if (info) {
			CHECK_FAST(fastGpuTimerStop(importFromHostTimer));
			CHECK_FAST(fastGpuTimerGetTime(importFromHostTimer, &elapsedTimeGpu));

			fullTime += elapsedTimeGpu;
			printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);

			CHECK_FAST(fastGpuTimerStart(unsharpMaskTimer));
		}

		fastNPPUnsharpMaskFilter_t param = { 0 };
		param.amount = static_cast<float>(options.NppImageFilter.Amount);
		param.sigma = static_cast<float>(options.NppImageFilter.Sigma);
		param.envelopSigma = static_cast<float>(options.NppImageFilter.envelopSigma);
		param.envelopMedian = static_cast<float>(options.NppImageFilter.envelopMedian);
		param.envelopCoef = static_cast<float>(options.NppImageFilter.envelopCoof);
		param.envelopRank = options.NppImageFilter.envelopRank;
		param.threshold = static_cast<float>(options.NppImageFilter.Threshold);

		CHECK_FAST(fastNppFilterTransform(
			hUnsharpMask,
			img.w,
			img.h,
			&param
		));

		if (info) {
			CHECK_FAST(fastGpuTimerStop(unsharpMaskTimer));
			CHECK_FAST(fastGpuTimerGetTime(unsharpMaskTimer, &elapsedTimeGpu));

			fullTime += elapsedTimeGpu;
			printf("Gauss filter transform time = %.2f ms\n\n", elapsedTimeGpu);

			CHECK_FAST(fastGpuTimerStart(exportToHostTimer));
		}

		fastExportParameters_t exportParameters = { };
		exportParameters.convert = options.ConvertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
		CHECK_FAST(fastExportToHostCopy(
			hExportToHost,

			h_Result.get(),
			img.w,
			img.wPitch,
			img.h,

			&exportParameters
		));

		if (info) {
			CHECK_FAST(fastGpuTimerStop(exportToHostTimer));
			CHECK_FAST(fastGpuTimerGetTime(exportToHostTimer, &elapsedTimeGpu));

			fullTime += elapsedTimeGpu;
			printf("Device-to-host transfer = %.2f ms\n\n", elapsedTimeGpu);
		}

		printf("Output image: %s\n\n", img.outputFileName.c_str());

		CHECK_FAST_SAVE_FILE(fvSaveImageToFile(
			(char *)img.outputFileName.c_str(),
			h_Result,
			surfaceFmt,
			img.h,
			img.w,
			img.wPitch,
			false
		));
	}

	if (info) {
		printf("Total time for all images = %.2f ms\n", fullTime);
		CHECK_FAST(fastGpuTimerDestroy(importFromHostTimer));
		CHECK_FAST(fastGpuTimerDestroy(unsharpMaskTimer));
		CHECK_FAST(fastGpuTimerDestroy(exportToHostTimer));
	}

	return FAST_OK;
}

fastStatus_t UnsharpMask::Close(void) const {
	CHECK_FAST(fastNppFilterDestroy(hUnsharpMask));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));
	CHECK_FAST(fastExportToHostDestroy(hExportToHost));

	return FAST_OK;
}
