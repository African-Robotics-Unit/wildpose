/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "stdio.h"
#include <list>

#include "DebayerMux.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "checks.h"

fastStatus_t DebayerMux::Init(DebayerSampleOptions &options, float *matrixA, char *matrixB) {
	convertToBGR = options.ConvertToBGR;
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;

	bayerPattern = options.Debayer.BayerFormat;
	muxBuffers[0] = muxBuffers[1] = NULL;

	CHECK_FAST(fastImportFromHostCreate(
		&hImportFromHost,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&muxBuffers[0]
	));

	if (matrixA != NULL || matrixB != NULL) {
		fastSam_t madParameter = { 0 };
		madParameter.correctionMatrix = matrixA;
		madParameter.blackShiftMatrix = matrixB;

		CHECK_FAST(fastImageFilterCreate(
			&hSam,

			FAST_SAM,
			(void *)&madParameter,

			options.MaxWidth,
			options.MaxHeight,

			muxBuffers[0],
			&muxBuffers[1]
		));
	}

	fastDeviceSurfaceBufferInfo_t info = { 0 };
	CHECK_FAST(fastGetDeviceSurfaceBufferInfo(muxBuffers[0], &info));
	printf("muxBuffer[0] parameters: surface format code = %d, width = %d, height = %d, pitch = %d\n", info.surfaceFmt, info.width, info.height, info.pitch);

	CHECK_FAST(fastGetDeviceSurfaceBufferInfo(muxBuffers[1], &info));
	printf("muxBuffer[1] parameters: surface format code = %d, width = %d, height = %d, pitch = %d\n", info.surfaceFmt, info.width, info.height, info.pitch);
	
	CHECK_FAST(fastMuxCreate(
		&hMux,

		muxBuffers,
		2,

		&ddebayerBuffer
	));

	CHECK_FAST(fastDebayerCreate(
		&hDebayer,

		options.Debayer.BayerType,

		options.MaxWidth,
		options.MaxHeight,

		ddebayerBuffer,
		&dstBuffer
	));

	CHECK_FAST(fastExportToHostCreate(
		&hExportToHost,

		&dstSurfaceFmt,

		dstBuffer
	));

	unsigned pitch = GetPitchFromSurface(dstSurfaceFmt, options.MaxWidth);

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(buffer.reset(static_cast<unsigned char *>(alloc.allocate(pitch * options.MaxHeight))));

	size_t requestedMemSpace = 0;
	size_t tmp;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;
	
	if (hSam != NULL) {
		CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hSam, &tmp));
		requestedMemSpace += tmp;
	}

	CHECK_FAST(fastDebayerGetAllocatedGpuMemorySize(hDebayer, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastExportToHostGetAllocatedGpuMemorySize(hExportToHost, &tmp));
	requestedMemSpace += tmp;
	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t DebayerMux::Transform(
	std::list< Image<FastAllocator> > &image
) {
	fastGpuTimerHandle_t hostToDeviceTimer = NULL;
	fastGpuTimerHandle_t madTimer = NULL;
	fastGpuTimerHandle_t debayerTimer = NULL;
	fastGpuTimerHandle_t deviceToHostTimer = NULL;

	float elapsedTimeGpu = 0.;
	float totalTime = 0.;

	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
		fastGpuTimerCreate(&madTimer);
		fastGpuTimerCreate(&debayerTimer);
		fastGpuTimerCreate(&deviceToHostTimer);
	}

	for (auto i = image.begin(); i != image.end(); ++i) {
		Image<FastAllocator> &img = *i;

		printf("Input image: %s\nImage size: %dx%d pixels\n\n", img.inputFileName.c_str(), img.w, img.h);
		printf("Output surface format: %s\n", EnumToString(dstSurfaceFmt));

		if (img.w > maxWidth ||
			img.h > maxHeight) {
			fprintf(stderr, "Unsupported image size\n");
			continue;
		}

		if (info) {
			fastGpuTimerStart(hostToDeviceTimer);
		}

		CHECK_FAST(fastImportFromHostCopy(
			hImportFromHost,

			img.data.get(),
			img.w,
			img.wPitch,
			img.h
		));

		if (info) {
			fastGpuTimerStop(hostToDeviceTimer);
			fastGpuTimerGetTime(hostToDeviceTimer, &elapsedTimeGpu);

			totalTime += elapsedTimeGpu;
			printf("Host-to-device transfer = %.2f ms\n", elapsedTimeGpu);
		}

		if (hSam != NULL) {
			if (info) {
				fastGpuTimerStart(madTimer);
			}

			CHECK_FAST(fastImageFiltersTransform(
				hSam,
				NULL,

				img.w,
				img.h
			));

			if (info) {
				fastGpuTimerStop(madTimer);
				fastGpuTimerGetTime(madTimer, &elapsedTimeGpu);

				totalTime += elapsedTimeGpu;
				printf("MAD time = %.2f ms\n", elapsedTimeGpu);
			}
		}

		CHECK_FAST(fastMuxSelect(hMux, hSam != NULL ? 1 : 0));

		if (info) {
			fastGpuTimerStart(debayerTimer);
		}

		CHECK_FAST(fastDebayerTransform(
			hDebayer,

			bayerPattern,

			img.w,
			img.h
		));

		if (info) {
			fastGpuTimerStop(debayerTimer);
			fastGpuTimerGetTime(debayerTimer, &elapsedTimeGpu);

			totalTime += elapsedTimeGpu;
			printf("Effective debayer performance = %.2f Gpixel/s (%.2f ms)\n\n", double(img.h * img.w) / elapsedTimeGpu * 1E-6, elapsedTimeGpu);

			fastGpuTimerStart(deviceToHostTimer);
		}

		fastExportParameters_t exportParameters = { };
		exportParameters.convert = convertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
		CHECK_FAST(fastExportToHostCopy(
			hExportToHost,

			buffer.get(),
			img.w,
			GetPitchFromSurface(dstSurfaceFmt, img.w),
			img.h,

			&exportParameters
		));

		if (info) {
			fastGpuTimerStop(deviceToHostTimer);
			fastGpuTimerGetTime(deviceToHostTimer, &elapsedTimeGpu);

			printf("Device-to-host transfer = %.2f ms\n", elapsedTimeGpu);
			totalTime += elapsedTimeGpu;
		}

		printf("Output image: %s\n\n", img.outputFileName.c_str());

		CHECK_FAST_SAVE_FILE(fvSaveImageToFile(
			(char *)img.outputFileName.c_str(),

			buffer,
			dstSurfaceFmt,
			img.h,
			img.w,
			GetPitchFromSurface(dstSurfaceFmt, img.w),
			false
		));
	}

	if (info) {
		printf("Total time for all images = %.2f ms\n", totalTime);
		fastGpuTimerDestroy(hostToDeviceTimer);
		fastGpuTimerDestroy(madTimer);
		fastGpuTimerDestroy(debayerTimer);
		fastGpuTimerDestroy(deviceToHostTimer);
	}

	return FAST_OK;
}

fastStatus_t DebayerMux::Close(void) {
	CHECK_FAST(fastExportToHostDestroy(hExportToHost));
	if (hSam != NULL) {
		CHECK_FAST(fastImageFiltersDestroy(hSam));
	}
	CHECK_FAST(fastDebayerDestroy(hDebayer));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));
	CHECK_FAST(fastMuxDestroy(hMux));
	
	return FAST_OK;
}
