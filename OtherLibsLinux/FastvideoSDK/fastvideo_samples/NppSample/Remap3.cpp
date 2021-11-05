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

#include "Remap3.h"
#include "supported_files.hpp"
#include "checks.h"

#include "fastvideo_sdk.h"

fastStatus_t Remap3::Init(NppRemapSampleOptions &options) {
	convertToBGR = options.ConvertToBGR;
	this->options = options;

	fastNPPRemapMap_t mapR = { 0 }, mapG = { 0 }, mapB = { 0 };
	{
		mapR.dstWidth = mapG.dstWidth = mapB.dstWidth = options.Remap.Rotate90 ? options.MaxHeight : options.MaxWidth;
		mapR.dstHeight = mapG.dstHeight = mapB.dstHeight = options.Remap.Rotate90 ? options.MaxWidth : options.MaxHeight;

		const unsigned mapPitch_ = uSnapUp<unsigned>(mapR.dstWidth, FAST_ALIGNMENT);

		CHECK_FAST(fastMalloc(reinterpret_cast<void**>(&mapR.mapX), mapPitch_ * mapR.dstHeight * sizeof(float)));
		CHECK_FAST(fastMalloc(reinterpret_cast<void**>(&mapR.mapY), mapPitch_ * mapR.dstHeight * sizeof(float)));
		CHECK_FAST(fastMalloc(reinterpret_cast<void**>(&mapG.mapX), mapPitch_ * mapR.dstHeight * sizeof(float)));
		CHECK_FAST(fastMalloc(reinterpret_cast<void**>(&mapG.mapY), mapPitch_ * mapR.dstHeight * sizeof(float)));
		CHECK_FAST(fastMalloc(reinterpret_cast<void**>(&mapB.mapX), mapPitch_ * mapR.dstHeight * sizeof(float)));
		CHECK_FAST(fastMalloc(reinterpret_cast<void**>(&mapB.mapY), mapPitch_ * mapR.dstHeight * sizeof(float)));

		if (options.Remap.Rotate90) {
			for (unsigned i = 0; i < mapR.dstHeight; i++) {
				for (unsigned j = 0; j < mapR.dstWidth; j++) {
					mapR.mapX[i * mapPitch_ + j] = mapG.mapX[i * mapPitch_ + j] = mapB.mapX[i * mapPitch_ + j] = static_cast<float>(mapR.dstHeight - i - 1);
					mapR.mapY[i * mapPitch_ + j] = mapG.mapY[i * mapPitch_ + j] = mapB.mapY[i * mapPitch_ + j] = static_cast<float>(mapR.dstWidth - j - 1);
				}
			}
		} else {
			for (unsigned i = 0; i < mapR.dstHeight; i++) {
				for (unsigned j = 0; j < mapR.dstWidth; j++) {
					mapR.mapX[i * mapPitch_ + j] = mapG.mapX[i * mapPitch_ + j] = mapB.mapX[i * mapPitch_ + j] = static_cast<float>(mapR.dstWidth - j - 1);
					mapR.mapY[i * mapPitch_ + j] = mapG.mapY[i * mapPitch_ + j] = mapB.mapY[i * mapPitch_ + j] = static_cast<float>(i - 1);
				}
			}
		}
	}

	fastNPPRemapBackground_t *background = NULL;
	if (options.Remap.BackgroundEnabled) {
		background = new fastNPPRemapBackground_t[1];
		background->isEnabled = false;

		background->R = options.Remap.BackgroundR;
		background->G = options.Remap.BackgroundG;
		background->B = options.Remap.BackgroundB;
		background->isEnabled = true;
	}

	fastNPPRemap3_t params = { 0 };
	{
		params.map[0] = &mapR;
		params.map[1] = &mapG;
		params.map[2] = &mapB;
		params.background = background;
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
		&hRemap,
		FAST_NPP_GEOMETRY_REMAP3,
		options.Interpolation.Type,

		&params,
		mapR.dstWidth,
		mapR.dstHeight,
		
		srcBuffer,
		&dstBuffer
	));

	CHECK_FAST(fastExportToHostCreate(
		&hExportToHost,

		&surfaceFmt,

		dstBuffer
	));

	const unsigned pitch = GetPitchFromSurface(surfaceFmt, std::max(options.MaxWidth, options.MaxHeight));

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(h_Result.reset(static_cast<unsigned char *>(alloc.allocate(pitch * std::max(options.MaxWidth, options.MaxHeight)))));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;
	CHECK_FAST(fastNppGeometryGetAllocatedGpuMemorySize(hRemap, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;
	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	CHECK_FAST(fastFree(mapR.mapX));
	CHECK_FAST(fastFree(mapR.mapY));
	CHECK_FAST(fastFree(mapG.mapX));
	CHECK_FAST(fastFree(mapG.mapY));
	CHECK_FAST(fastFree(mapB.mapX));
	CHECK_FAST(fastFree(mapB.mapY));

	if (background != NULL) {
		delete[] background;
		background = NULL;
	}

	return FAST_OK;
}

fastStatus_t Remap3::Transform(std::list< Image<FastAllocator> > &image) {
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
			hRemap,
			NULL,
			img.w, img.h
		));

		if (info) {
			fastGpuTimerStop(remapTimer);
			fastGpuTimerGetTime(remapTimer, &elapsedTimeGpu);

			fullTime += elapsedTimeGpu;
			printf("Remap transform time = %.2f ms\n\n", elapsedTimeGpu);
		}

		if (info) {
			fastGpuTimerStart(exportToHostTimer);
		}

		fastExportParameters_t exportParameters = { };
		exportParameters.convert = convertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;

		const int dstWidth = options.Remap.Rotate90 ? img.h : img.w;
		const int dstHeight = options.Remap.Rotate90 ? img.w : img.h;

		CHECK_FAST(fastExportToHostCopy(
			hExportToHost,

			h_Result.get(),
			dstWidth,
			GetPitchFromSurface(surfaceFmt, dstWidth),
			dstHeight,

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
			dstHeight,
			dstWidth,
			GetPitchFromSurface(surfaceFmt, dstWidth),
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

fastStatus_t Remap3::Close(void) const {
	CHECK_FAST(fastNppGeometryDestroy(hRemap));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));
	CHECK_FAST(fastExportToHostDestroy(hExportToHost));

	return FAST_OK;
}
