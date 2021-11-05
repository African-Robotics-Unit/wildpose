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

#include "SDIImportFromHost.h"
#include "checks.h"
#include "supported_files.hpp"
#include "HelperSDI.hpp"

fastStatus_t SDIImportFromHost::Init(SDIConverterSampleOptions &options) {
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;
	sdiFmt = options.SDI.SDIFormat;
	convertToBGR = options.ConvertToBGR;

	fastSDIRaw12Import_t convert16;
	convert16.isConvert12to16 = options.SDI.IsConvert12to16;

	CHECK_FAST(fastSDIImportFromHostCreate(
		&hImport,

		options.SDI.SDIFormat,
		&convert16,

		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	CHECK_FAST(fastExportToHostCreate(
		&hExportToHost,

		&surfaceFmt,

		srcBuffer
	));

	maxPitch = GetPitchFromSurface(surfaceFmt, options.MaxWidth);
	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(h_Result.reset((unsigned char *)alloc.allocate(maxPitch * options.MaxHeight)));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;

	CHECK_FAST(fastSDIImportFromHostGetAllocatedGpuMemorySize(hImport, &tmp));
	requestedMemSpace += tmp;

	CHECK_FAST(fastExportToHostGetAllocatedGpuMemorySize(hExportToHost, &tmp));
	requestedMemSpace += tmp;

	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t SDIImportFromHost::Transform(Image<FastAllocator > &img, char *outFilename) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t exportToHostTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&importFromHostTimer);
		fastGpuTimerCreate(&exportToHostTimer);

		fastGpuTimerStart(importFromHostTimer);
	}

	if (
		sdiFmt == FAST_SDI_422_10_CbYCrY_PACKED_BT601 ||
		sdiFmt == FAST_SDI_422_10_CbYCrY_PACKED_BT601_FR ||
		sdiFmt == FAST_SDI_422_10_CbYCrY_PACKED_BT709 ||
		sdiFmt == FAST_SDI_422_10_CbYCrY_PACKED_BT2020
	) {
		CHECK_FAST(fastSDIImportFromHostCopyPacked(
			hImport,

			img.data.get(),
			GetSDIPitch(sdiFmt, img.w),

			img.w,
			img.h
		));
	} else {
		CHECK_FAST(fastSDIImportFromHostCopy(
			hImport,

			img.data.get(),

			img.w,
			img.h
		));
	}

	if (info) {
		fastGpuTimerStop(importFromHostTimer);
		fastGpuTimerGetTime(importFromHostTimer, &elapsedTimeGpu);

		fullTime += elapsedTimeGpu;
		printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);

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

	printf("Output image: %s\n\n", outFilename);

	CHECK_FAST_SAVE_FILE(fvSaveImageToFile(
		outFilename,
		h_Result,
		surfaceFmt,
		img.h,
		img.w,
		GetPitchFromSurface(surfaceFmt, img.w),
		false
	));

	if (info) {
		fastGpuTimerDestroy(importFromHostTimer);
		fastGpuTimerDestroy(exportToHostTimer);
	}

	return FAST_OK;
}

fastStatus_t SDIImportFromHost::Transform3(Image<FastAllocator > &img, char *outFilename) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t exportToHostTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&importFromHostTimer);
		fastGpuTimerCreate(&exportToHostTimer);
	}

	fastChannelDescription_t srcY = { 0 }, srcU = { 0 }, srcV = { 0 };
	// Y
	{
		srcY.data = static_cast<unsigned char *>(img.data.get());
		srcY.width =  img.w;
		srcY.height = img.h;
		srcY.pitch = GetSDIPitchY(sdiFmt, img.w);
	}

	// U
	{
		srcU.data = &srcY.data[srcY.pitch * srcY.height];
		srcU.width = GetSDIWidthUV(sdiFmt, img.w);
		srcU.height = GetSDIHeightUV(sdiFmt, img.h);
		srcU.pitch = GetSDIPitchUV(sdiFmt, img.w);
	}

	// V
	{
		srcV.data = &srcU.data[srcU.pitch * srcU.height];
		srcV.width = srcU.width;
		srcV.height = srcU.height;
		srcV.pitch = srcU.pitch;
	}

	if (info) {
		fastGpuTimerStart(importFromHostTimer);
	}

	CHECK_FAST(fastSDIImportFromHostCopy3(
		hImport,

		&srcY,
		&srcU,
		&srcV
	));

	if (info) {
		fastGpuTimerStop(importFromHostTimer);
		fastGpuTimerGetTime(importFromHostTimer, &elapsedTimeGpu);

		fullTime += elapsedTimeGpu;
		printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);

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

	printf("Output image: %s\n\n", outFilename);

	CHECK_FAST_SAVE_FILE(fvSaveImageToFile(
		outFilename,
		h_Result,
		surfaceFmt,
		img.h,
		img.w,
		GetPitchFromSurface(surfaceFmt, img.w),
		false
	));

	if (info) {
		fastGpuTimerDestroy(importFromHostTimer);
		fastGpuTimerDestroy(exportToHostTimer);
	}

	return FAST_OK;
}

fastStatus_t SDIImportFromHost::Close(void) const {
	CHECK_FAST(fastSDIImportFromHostDestroy(hImport));
	CHECK_FAST(fastExportToHostDestroy(hExportToHost));

	return FAST_OK;
}
