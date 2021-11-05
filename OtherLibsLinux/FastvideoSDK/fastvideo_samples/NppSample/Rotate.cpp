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
#include <cmath>

#include "Rotate.h"
#include "supported_files.hpp"
#include "checks.h"

#include "fastvideo_sdk.h"

fastStatus_t Rotate::Init(NppRotateSampleOptions &options) {
	convertToBGR = options.ConvertToBGR;
	this->options = options;

	CHECK_FAST(fastGetSdkParametersHandle(&hSdkParameters));
	CHECK_FAST(fastImportFromHostCreate(
		&hImportFromHost,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	CHECK_FAST(fastNppRotateLibraryInit(hSdkParameters));
	CHECK_FAST(fastNppRotateCreate(
		&hRotate,
		options.Interpolation.Type,
		
		srcBuffer,
		&dstBuffer
	));

	fastDeviceSurfaceBufferInfo_t bufferInfo = { 0 };

	CHECK_FAST(fastGetDeviceSurfaceBufferInfo(
		dstBuffer,
		&bufferInfo
	));

	CHECK_FAST(fastExportToHostCreate(
		&hExportToHost,

		&surfaceFmt,

		dstBuffer
	));

	// sqrt(options.MaxWidth * options.MaxWidth + options.MaxHeight * options.MaxHeight);
	const unsigned maxWidth = bufferInfo.maxWidth;
	const unsigned maxHeight = bufferInfo.maxHeight;

	unsigned pitch = GetPitchFromSurface(surfaceFmt, maxWidth);

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(h_Result.reset(static_cast<unsigned char *>(alloc.allocate(pitch * maxHeight))));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;
	CHECK_FAST(fastNppRotateGetAllocatedGpuMemorySize(hRotate, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;
	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t Rotate::Transform(
	std::list< Image<FastAllocator> > &image
) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t rotateTimer = NULL;
	fastGpuTimerHandle_t exportToHostTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&importFromHostTimer);
		fastGpuTimerCreate(&rotateTimer);
		fastGpuTimerCreate(&exportToHostTimer);
	}

	for (auto i = image.begin(); i != image.end(); i++) {
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

		NppQuadCorners_t quadCorners = { 0 };
		CHECK_FAST(fastNppRotateGetRotateQuad(
			hRotate,

			img.w,
			img.h,

			options.Rotate.Angle,

			options.Rotate.Shift,
			options.Rotate.Shift,

			&quadCorners
		));
		
		const unsigned maxSize = static_cast<unsigned>(sqrt(options.MaxWidth * options.MaxWidth + options.MaxHeight * options.MaxHeight));
		unsigned rotateWidth;
		double shiftX;
		{
			shiftX = quadCorners.leftTopCorner.x;
			shiftX =  std::min(quadCorners.leftBottomCorner.x, shiftX);
			shiftX = std::min(quadCorners.rightTopCorner.x, shiftX);
			shiftX = std::min(quadCorners.rightBottomCorner.x, shiftX);

			double maxX = quadCorners.leftTopCorner.x;
			maxX = std::max(quadCorners.leftBottomCorner.x, maxX);
			maxX = std::max(quadCorners.rightTopCorner.x, maxX);
			maxX = std::max(quadCorners.rightBottomCorner.x, maxX);

			rotateWidth = abs(static_cast<int>(maxX - shiftX + 0.5));
		}

		unsigned rotateHeight;
		double shiftY;
		{
			shiftY = quadCorners.leftTopCorner.y;
			shiftY =  std::min(quadCorners.leftBottomCorner.y, shiftY);
			shiftY = std::min(quadCorners.rightTopCorner.y, shiftY);
			shiftY = std::min(quadCorners.rightBottomCorner.y, shiftY);

			double maxX = quadCorners.leftTopCorner.y;
			maxX = std::max(quadCorners.leftBottomCorner.y, maxX);
			maxX = std::max(quadCorners.rightTopCorner.y, maxX);
			maxX = std::max(quadCorners.rightBottomCorner.y, maxX);

			rotateHeight = abs(static_cast<int>(maxX - shiftY + 0.5));
		}
		
		if (info) {
			fastGpuTimerStart(rotateTimer);
		}

		CHECK_FAST(fastNppRotateTransform(
			hRotate,

			img.w,
			img.h,

			rotateWidth, // dstRoiX
			rotateHeight, // dstRoiY

			options.Rotate.Angle,
				
			abs(static_cast<int>(shiftX + 0.5)),
			abs(static_cast<int>(shiftY + 0.5))
		));

		if (info) {
			fastGpuTimerStop(rotateTimer);
			fastGpuTimerGetTime(rotateTimer, &elapsedTimeGpu);

			fullTime += elapsedTimeGpu;
			printf("Rotate transform time = %.2f ms\n\n", elapsedTimeGpu);
		}

		if (info) {
			fastGpuTimerStart(exportToHostTimer);
		}

		fastExportParameters_t exportParameters = { };
		exportParameters.convert = convertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
		CHECK_FAST(fastExportToHostCopy(
			hExportToHost,

			h_Result.get(),
			rotateWidth,
			GetPitchFromSurface(surfaceFmt, rotateWidth),
			rotateHeight,

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
			rotateHeight,
			rotateWidth,
			GetPitchFromSurface(surfaceFmt, rotateWidth),
			false
		));
	}

	if (info) {
		printf("Total time for all images = %.2f ms\n", fullTime);
		fastGpuTimerDestroy(importFromHostTimer);
		fastGpuTimerDestroy(rotateTimer);
		fastGpuTimerDestroy(exportToHostTimer);
	}

	return FAST_OK;
}

fastStatus_t Rotate::Close(void) const {
	CHECK_FAST(fastNppRotateDestroy(hRotate));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));
	CHECK_FAST(fastExportToHostDestroy(hExportToHost));

	return FAST_OK;
}
