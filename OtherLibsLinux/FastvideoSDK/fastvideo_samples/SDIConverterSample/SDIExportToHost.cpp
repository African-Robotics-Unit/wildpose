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

#include "SDIExportToHost.h"
#include "checks.h"

#include "HelperSDI.hpp"
#include "SDICommon.hpp"

fastStatus_t SDIExportToHost::Init(SDIConverterSampleOptions &options) {
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;

	sdiFmt = options.SDI.SDIFormat;

	void *staticParam = nullptr;

	if (sdiFmt == FAST_SDI_NV12_BT601_FR || sdiFmt == FAST_SDI_P010_BT601_FR) { //should be all 420 formats
		if ((maxWidth & 1) != 0 || (maxHeight & 1) != 0) {
			fprintf(stderr, "Unsupported max image size\n");
			return FAST_INVALID_SIZE;
		}
	}
	if (sdiFmt == FAST_SDI_RGBA) {
		staticParam = new fastSDIRGBAExport_t();
		static_cast<fastSDIRGBAExport_t*>(staticParam)->padding = options.SDI.alphaPadding;
	}

	CHECK_FAST(fastImportFromHostCreate(
		&hImportFromHost,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	CHECK_FAST(fastSDIExportToHostCreate(
		&hExport,

		options.SDI.SDIFormat,
		staticParam,

		options.MaxWidth,
		options.MaxHeight,

		srcBuffer
	));

	if (sdiFmt == FAST_SDI_RGBA) {
		delete staticParam;
	}
	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(h_Result.reset(static_cast<unsigned char *>(alloc.allocate(GetSDIBufferSize(sdiFmt, maxWidth, maxHeight)))));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;

	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;

	CHECK_FAST(fastSDIExportToHostGetAllocatedGpuMemorySize(hExport, &tmp));
	requestedMemSpace += tmp;

	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}



fastStatus_t SDIExportToHost::Transform(
	Image<FastAllocator > &img,
	char *outFilename) const
{
	if (img.w > maxWidth ||
		img.h > maxHeight) {
		fprintf(stderr, "Unsupported image size\n");
		return FAST_INVALID_SIZE;
	}
	if (sdiFmt == FAST_SDI_NV12_BT601_FR || sdiFmt == FAST_SDI_P010_BT601_FR) {
		if ((img.w & 1) != 0 || (img.h & 1) != 0) {
			fprintf(stderr, "Unsupported image size\n");
			return FAST_INVALID_SIZE;
		}
	}

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t exportToHostTimer = NULL;
	if (info) {
		fastGpuTimerCreate(&importFromHostTimer);
		fastGpuTimerCreate(&exportToHostTimer);

		fastGpuTimerStart(importFromHostTimer);
	}

	CHECK_FAST(fastImportFromHostCopy(
		hImportFromHost,

		img.data.get(),
		img.w,
		img.wPitch,
		img.h
	));

	float elapsedTimeGpu = 0.;
	if (info) {
		fastGpuTimerStop(importFromHostTimer);
		fastGpuTimerGetTime(importFromHostTimer, &elapsedTimeGpu);

		printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);

		fastGpuTimerStart(exportToHostTimer);
	}

	unsigned outputWidth = img.w, outputHeight = img.h;
	CHECK_FAST(fastSDIExportToHostCopy(
		hExport,

		h_Result.get(),

		&outputWidth,
		&outputHeight
	));

	if (info) {
		fastGpuTimerStop(exportToHostTimer);
		fastGpuTimerGetTime(exportToHostTimer, &elapsedTimeGpu);

		printf("Device-to-host transfer = %.2f ms\n\n", elapsedTimeGpu);

		fastGpuTimerDestroy(importFromHostTimer);
		fastGpuTimerDestroy(exportToHostTimer);
	}

	printf("Output file: %s\n\n", outFilename);
	CHECK_FAST(fvSaveBinary(outFilename, h_Result.get(), GetSDIBufferSize(sdiFmt, outputWidth, outputHeight)));

	return FAST_OK;
}


fastStatus_t SDIExportToHost::Transform3(
	Image<FastAllocator >& img,
	char* outFilename) const
{
	if (img.w > maxWidth ||
		img.h > maxHeight) {
		fprintf(stderr, "Unsupported image size\n");
		return FAST_INVALID_SIZE;
	}
	if (sdiFmt == FAST_SDI_NV12_BT601_FR || sdiFmt == FAST_SDI_P010_BT601_FR) {
		if ((img.w & 1) != 0 || (img.h & 1) != 0) {
			fprintf(stderr, "Unsupported image size\n");
			return FAST_INVALID_SIZE;
		}
	}

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t exportToHostTimer = NULL;

	unsigned outputWidth = img.w, outputHeight = img.h;
	memset(h_Result.get(), 0, GetSDIBufferSize(sdiFmt, outputWidth, outputHeight));

	if (info) {
		fastGpuTimerCreate(&importFromHostTimer);
		fastGpuTimerCreate(&exportToHostTimer);

		fastGpuTimerStart(importFromHostTimer);
	}

	CHECK_FAST(fastImportFromHostCopy(
		hImportFromHost,

		img.data.get(),
		img.w,
		img.wPitch,
		img.h
	));

	float elapsedTimeGpu = 0.;
	if (info) {
		fastGpuTimerStop(importFromHostTimer);
		fastGpuTimerGetTime(importFromHostTimer, &elapsedTimeGpu);

		printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);

		fastGpuTimerStart(exportToHostTimer);
	}

	
		fastChannelDescription_t dstY = { 0 }, dstU = { 0 }, dstV = { 0 };

		dstY.data = h_Result.get();
		dstY.pitch = GetSDIPitchY(sdiFmt, img.w);

		dstU.data = dstY.data + dstY.pitch * img.h;
		dstU.pitch = GetSDIPitchUV(sdiFmt, img.w);

		dstV.data = dstU.data + dstU.pitch * GetSDIHeightUV(sdiFmt, img.h);
		dstV.pitch = GetSDIPitchUV(sdiFmt, img.w);

		CHECK_FAST(fastSDIExportToHostCopy3(
			hExport,
			&dstY,
			&dstU,
			&dstV
		));

	if (info) {
		fastGpuTimerStop(exportToHostTimer);
		fastGpuTimerGetTime(exportToHostTimer, &elapsedTimeGpu);

		printf("Device-to-host transfer = %.2f ms\n\n", elapsedTimeGpu);

		fastGpuTimerDestroy(importFromHostTimer);
		fastGpuTimerDestroy(exportToHostTimer);
	}

	printf("Output file: %s\n\n", outFilename);
	CHECK_FAST(fvSaveBinary(outFilename, h_Result.get(), GetSDIBufferSize(sdiFmt, outputWidth, outputHeight)));
	return FAST_OK;
}

fastStatus_t SDIExportToHost::Close(void) const {
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));
	CHECK_FAST(fastSDIExportToHostDestroy(hExport));

	return FAST_OK;
}
