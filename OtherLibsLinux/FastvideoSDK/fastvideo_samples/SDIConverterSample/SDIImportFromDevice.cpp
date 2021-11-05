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

#include "SDIImportFromDevice.h"
#include "HelperSDI.hpp"
#include "checks.h"
#include "supported_files.hpp"
#include <cuda_runtime.h>

fastStatus_t SDIImportFromDevice::Init(SDIConverterSampleOptions &options) {
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;
	sdiFmt = options.SDI.SDIFormat;
	convertToBGR = options.ConvertToBGR;

	fastSDIRaw12Import_t convert16;
	convert16.isConvert12to16 = options.SDI.IsConvert12to16;

	CHECK_FAST(fastSDIImportFromDeviceCreate(
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
	CHECK_FAST_ALLOCATION(h_src.reset((unsigned char *)alloc.allocate(GetDeviceSDIBufferSize(sdiFmt, maxWidth, maxHeight))));

	const cudaError_t error = cudaMalloc(&d_srcUnpacked, GetDeviceSDIBufferSize(sdiFmt, maxWidth, maxHeight));
	if (error != cudaSuccess) {
		fprintf(stderr, "GPU memory allocation failed: %s\n", cudaGetErrorString(error));
		return FAST_INSUFFICIENT_DEVICE_MEMORY;
	}

	size_t requestedMemSpace = 0;
	size_t tmp = 0;

	CHECK_FAST(fastSDIImportFromDeviceGetAllocatedGpuMemorySize(hImport, &tmp));
	requestedMemSpace += tmp;

	CHECK_FAST(fastExportToHostGetAllocatedGpuMemorySize(hExportToHost, &tmp));
	requestedMemSpace += tmp;

	printf("\nRequested GPU memory space in SDK: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t SDIImportFromDevice::Transform(Image<FastAllocator > &img, char *outFilename) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t exportToHostTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&importFromHostTimer);
		fastGpuTimerCreate(&exportToHostTimer);

		fastGpuTimerStart(importFromHostTimer);
	}

	UnPackDeviceSDI(img.data.get(), h_src.get(), sdiFmt, img.w, img.h);

	const cudaError_t error = cudaMemcpy(d_srcUnpacked, h_src.get(), GetDeviceSDIBufferSize(sdiFmt, img.w, img.h), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "GPU memory copy failed: %s\n", cudaGetErrorString(error));
		return FAST_INSUFFICIENT_DEVICE_MEMORY;
	}

	CHECK_FAST(fastSDIImportFromDeviceCopy(
		hImport,

		d_srcUnpacked,

		img.w,
		img.h
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

fastStatus_t SDIImportFromDevice::Transform3(Image<FastAllocator > &img, char *outFilename) {
	float elapsedTimeGpu = 0.;

	if (!IsSDICopy3Format(sdiFmt)) {
		fprintf(stderr, "Unsupported SDI format\n");
		return FAST_INVALID_VALUE;
	}

	fastGpuTimerHandle_t importFromHostTimer = NULL;
	fastGpuTimerHandle_t exportToHostTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&importFromHostTimer);
		fastGpuTimerCreate(&exportToHostTimer);
	}

	UnPackDeviceSDI(img.data.get(), h_src.get(), sdiFmt, img.w, img.h);

	cudaError_t error = cudaMemcpy(d_srcUnpacked, h_src.get(), GetSDIBufferSize(sdiFmt, img.w, img.h), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "GPU memory copy failed: %s\n", cudaGetErrorString(error));
		return FAST_INSUFFICIENT_DEVICE_MEMORY;
	}

	fastChannelDescription_t srcY = { 0 }, srcU = { 0 }, srcV = { 0 };
	// Y
	{
		srcY.data = static_cast<unsigned char *>(d_srcUnpacked);
		srcY.width = img.w;
		srcY.pitch = GetDeviceSDIPitchY(sdiFmt, img.w);
		srcY.height = img.h;
	}

	// U
	{
		srcU.data = &static_cast<unsigned char *>(d_srcUnpacked)[srcY.pitch * img.h];
		srcU.width = GetSDIWidthUV(sdiFmt, img.w);
		srcU.pitch = GetDeviceSDIPitchUV(sdiFmt, img.w);
		srcU.height = GetSDIHeightUV(sdiFmt, img.h);
	}

	// V
	{
		srcV.data = &static_cast<unsigned char *>(srcU.data)[srcU.pitch * srcU.height];
		srcV.width = srcU.width;
		srcV.pitch = srcU.pitch;
		srcV.height = srcU.height;
	}

	if (info) {
		fastGpuTimerStart(importFromHostTimer);
	}

	CHECK_FAST(fastSDIImportFromDeviceCopy3(
		hImport,

		&srcY,
		&srcU,
		&srcV
	));

	if (info) {
		fastGpuTimerStop(importFromHostTimer);
		fastGpuTimerGetTime(importFromHostTimer, &elapsedTimeGpu);

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

		printf("Device-to-host transfer = %.2f ms\n\n", elapsedTimeGpu);

		fastGpuTimerDestroy(importFromHostTimer);
		fastGpuTimerDestroy(exportToHostTimer);
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

	return FAST_OK;
}

fastStatus_t SDIImportFromDevice::Close(void) const {
	CHECK_FAST(fastSDIImportFromDeviceDestroy(hImport));
	CHECK_FAST(fastExportToHostDestroy(hExportToHost));

	if (d_srcUnpacked != NULL) {
		cudaError_t error = cudaFree(d_srcUnpacked);
		if (error != cudaSuccess) {
			fprintf(stderr, "GPU memory free failed: %s\n", cudaGetErrorString(error));
			return FAST_INSUFFICIENT_DEVICE_MEMORY;
		}
	}

	return FAST_OK;
}
