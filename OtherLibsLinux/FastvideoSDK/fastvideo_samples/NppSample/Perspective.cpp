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

#include "Perspective.h"
#include "supported_files.hpp"
#include "checks.h"

#include "fastvideo_sdk.h"

fastStatus_t Perspective::Init(NppPerspectiveSampleOptions &options) {
	convertToBGR = options.ConvertToBGR;
	this->options = options;

	fastNPPPerspective_t params = { 0 };
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
				params.coeffs[i][j] = options.Perspective.Coeffs[i][j];
			}
	}

	CHECK_FAST(fastGetSdkParametersHandle(&hSdkParameters));
	CHECK_FAST(fastImportFromHostCreate(
		&hImportFromHost,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	CHECK_FAST(fastNppGeometryLibraryInit(hSdkParameters));
	CHECK_FAST(fastNppGeometryCreate(
		&hPerspective,
		FAST_NPP_GEOMETRY_PERSPECTIVE,
		options.Interpolation.Type,

		&params,
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
	CHECK_FAST(fastNppGeometryGetAllocatedGpuMemorySize(hPerspective, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;
	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t Perspective::Transform(std::list< Image<FastAllocator> > &image) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t remapTimer = NULL;
	fastGpuTimerHandle_t exportToHostTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&importFromHostTimer);
		fastGpuTimerCreate(&remapTimer);
		fastGpuTimerCreate(&exportToHostTimer);
	}

	for (auto i = image.begin(); i != image.end(); ++i) {
		Image<FastAllocator> &img = *i;

		printf("Input image: %s\nImage size: %dx%d pixels\n\n", img.inputFileName.c_str(), img.w, img.h);
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
		}

		if(info) {
			fastGpuTimerStart(remapTimer);
		}

		CHECK_FAST(fastNppGeometryTransform(
			hPerspective,
			NULL,
			img.w, img.h
		));

		if (info) {
			fastGpuTimerStop(remapTimer);
			fastGpuTimerGetTime(remapTimer, &elapsedTimeGpu);

			fullTime += elapsedTimeGpu;
			printf("Perspective transform time = %.2f ms\n\n", elapsedTimeGpu);
		}

		if (info) {
			fastGpuTimerStart(exportToHostTimer);
		}

		fastExportParameters_t exportParameters = { };
		exportParameters.convert = convertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;

		CHECK_FAST(fastExportToHostCopy(
			hExportToHost,

			h_Result.get(),
			img.w,
			GetPitchFromSurface(surfaceFmt, img.w),
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
			const_cast<char *>(img.outputFileName.c_str()),
			h_Result,
			surfaceFmt,
			img.h,
			img.w,
			GetPitchFromSurface(surfaceFmt, img.w),
			false
		));
	}

	if (info) {
		printf("Total time for all images = %.2f ms\n", fullTime);
		fastGpuTimerDestroy(importFromHostTimer);
		fastGpuTimerDestroy(remapTimer);
		fastGpuTimerDestroy(exportToHostTimer);
	}

	return FAST_OK;
}

fastStatus_t Perspective::Close() const {
	CHECK_FAST(fastNppGeometryDestroy(hPerspective));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));
	CHECK_FAST(fastExportToHostDestroy(hExportToHost));

	return FAST_OK;
}
