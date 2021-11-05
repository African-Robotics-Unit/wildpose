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

#include "Sharpen.h"
#include "checks.h"
#include "supported_files.hpp"

fastStatus_t Sharp::Init(BaseOptions &options, bool IsSharpenFilter) {
	fastStatus_t ret;

	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;
	convertToBGR = options.ConvertToBGR;

	CHECK_FAST(fastImportFromHostCreate(
		&hImportFromHost,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	CHECK_FAST( fastImageFilterCreate(
		&hImageFilter,

		IsSharpenFilter?FAST_GAUSSIAN_SHARPEN: FAST_GAUSSIAN_BLUR,
		NULL,

		options.MaxWidth,
		options.MaxHeight,

		srcBuffer,
		&d_imageFilterBuffer
	) );


	CHECK_FAST(fastExportToHostCreate(
		&hExportToHost,

		&surfaceFmt,

		d_imageFilterBuffer
	));

	maxPitch = GetNumberOfChannelsFromSurface(options.SurfaceFmt) * (((options.MaxWidth * 2 + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT) * FAST_ALIGNMENT);

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(h_Result.reset((unsigned char *)alloc.allocate(maxPitch * options.MaxHeight)));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;

	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;

	CHECK_FAST( fastImageFiltersGetAllocatedGpuMemorySize( hImageFilter, &tmp ) );
	requestedMemSpace += tmp;

	CHECK_FAST(fastExportToHostGetAllocatedGpuMemorySize(hExportToHost, &tmp));
	requestedMemSpace += tmp;

	printf("Requested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t Sharp::Close() {
	fastStatus_t ret;

	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));
	CHECK_FAST(fastExportToHostDestroy(hExportToHost));
	CHECK_FAST(fastImageFiltersDestroy(hImageFilter));

	return FAST_OK;
}

fastStatus_t Sharp::Transform(std::list< Image< FastAllocator > > &image, double sigma) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t exportToHostTimer = NULL;
	fastGpuTimerHandle_t imageFilterTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&importFromHostTimer);
		fastGpuTimerCreate(&exportToHostTimer);
		fastGpuTimerCreate(&imageFilterTimer);
	}

	for (std::list<Image< FastAllocator > >::iterator i = image.begin(); i != image.end(); i++) {
		Image< FastAllocator> &img = *i;

		printf("Input image: %s\nImage size: %dx%d pixels\n\n", img.inputFileName.c_str(), img.w, img.h);
		printf("Sigma coefficient: %.3f\n", sigma);

		if (img.w > maxWidth ||
			img.h > maxHeight) {
			fprintf(stderr, "Unsupported image size\n");
			continue;
		}

		if (info) {
			fastGpuTimerStart(importFromHostTimer);
		}

		fastStatus_t ret;
		CHECK_FAST(fastImportFromHostCopy(
			hImportFromHost,

			img.data.get(),
			img.w,
			GetNumberOfChannelsFromSurface(surfaceFmt) * (((img.w * sizeof(unsigned char) + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT) * FAST_ALIGNMENT),
			img.h
		));

		if (info) {
			fastGpuTimerStop(importFromHostTimer);
			fastGpuTimerGetTime(importFromHostTimer, &elapsedTimeGpu);

			fullTime += elapsedTimeGpu;
			printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);

		}

		fastGaussianFilter_t gaussParameters;
		gaussParameters.sigma = sigma;

		if (info) {
			fastGpuTimerStart(imageFilterTimer);
		}

		if (fastImageFiltersTransform(
			hImageFilter,
			&gaussParameters,

			img.w,
			img.h
		) != FAST_OK) {
			fprintf(stderr, "Image filter transform failed (file %s)\n", (*i).inputFileName.c_str());
			continue;
		}

		if (info) {
			fastGpuTimerStop(imageFilterTimer);

			fastGpuTimerGetTime(imageFilterTimer, &elapsedTimeGpu);
			printf("Sharpen filter (before resize) time = %.2f ms\n", elapsedTimeGpu);

			fastGpuTimerStart(exportToHostTimer);
		}

		fastExportParameters_t exportParameters;
		exportParameters.convert = convertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
		CHECK_FAST(fastExportToHostCopy(
			hExportToHost,

			h_Result.get(),
			img.w,
			GetNumberOfChannelsFromSurface(surfaceFmt) * (((img.w * 2 + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT) * FAST_ALIGNMENT),
			img.h,

			&exportParameters
		));

		if (info) {
			fastGpuTimerStop(exportToHostTimer);
			fastGpuTimerGetTime(exportToHostTimer, &elapsedTimeGpu);

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
			GetNumberOfChannelsFromSurface(surfaceFmt) * (((img.w * 2 + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT) * FAST_ALIGNMENT),
			false
		));
	}

	if ( info ) {
		fastGpuTimerDestroy(importFromHostTimer);
		fastGpuTimerDestroy(exportToHostTimer);
		fastGpuTimerDestroy(imageFilterTimer);
	}

	return FAST_OK;
}
