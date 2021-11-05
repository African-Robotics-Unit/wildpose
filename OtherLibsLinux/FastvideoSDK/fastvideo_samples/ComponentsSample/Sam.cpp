/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <sstream>
#include <cstdio>

#include "Sam.h"
#include "supported_files.hpp"
#include "checks.h"
#include "BaseOptions.h"
#include "SurfaceTraits.hpp"


void Sam::PopulateLinerLut(unsigned short *lut, unsigned lutSize, unsigned newMaxValue)
{
	for (unsigned i = 0; i < lutSize; i++)
	{
		lut[i] = newMaxValue * i / lutSize;
	}
}

fastStatus_t Sam::Init(BaseOptions &options, float *matrixA, void *matrixB) {
	folder = options.IsFolder;
	convertToBGR = options.ConvertToBGR;
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;

	CHECK_FAST(fastImportFromHostCreate(
		&hHostToDevice,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	if (isTwoByteBlackShiftMatrix)
	{
		fastSam16_t madParameter = { 0 };
		madParameter.correctionMatrix = matrixA;
		madParameter.blackShiftMatrix = (short *)matrixB;

		CHECK_FAST(fastImageFilterCreate(
			&hSam,

			FAST_SAM16,
			static_cast<void *>(&madParameter),

			options.MaxWidth,
			options.MaxHeight,

			srcBuffer,
			&madBuffer
		));
	}
	else
	{
		fastSam_t madParameter = { 0 };
		madParameter.correctionMatrix = matrixA;
		madParameter.blackShiftMatrix = (char *)matrixB;

		CHECK_FAST(fastImageFilterCreate(
			&hSam,

			FAST_SAM,
			static_cast<void *>(&madParameter),

			options.MaxWidth,
			options.MaxHeight,

			srcBuffer,
			&madBuffer
		));
	}


	fastDeviceSurfaceBufferHandle_t *tmpBuffer = &madBuffer;
	if (options.SurfaceFmt == FAST_I10)
	{
		fastLut_10_t linerLut10 = { 0 };
		PopulateLinerLut(linerLut10.lut, 1024, 65535);
		CHECK_FAST(fastImageFilterCreate(
			&hLut,
			FAST_LUT_10_16,
			static_cast<void *>(&linerLut10),

			options.MaxWidth,
			options.MaxHeight,

			madBuffer,
			tmpBuffer
		));
	}
	else if (options.SurfaceFmt == FAST_I12)
	{
		fastLut_12_t linerLut12 = { 0 };
		PopulateLinerLut(linerLut12.lut, 4096, 65535);

		CHECK_FAST(fastImageFilterCreate(
			&hLut,
			FAST_LUT_12_16,
			static_cast<void *>(&linerLut12),

			options.MaxWidth,
			options.MaxHeight,

			madBuffer,
			tmpBuffer
		));
	}
	else if (options.SurfaceFmt == FAST_I14)
	{
		fastLut_16_t linerLut14 = { 0 };
		PopulateLinerLut(linerLut14.lut, 16384, 65535);

		CHECK_FAST(fastImageFilterCreate(
			&hLut,
			FAST_LUT_14_16,
			static_cast<void *>(&linerLut14),

			options.MaxWidth,
			options.MaxHeight,

			madBuffer,
			tmpBuffer
		));
	}

	CHECK_FAST(fastExportToHostCreate(
		&hDeviceToHost,
		&surfaceFmt,
		*tmpBuffer
	));

	FastAllocator allocator;
	unsigned pitch = GetPitchFromSurface(surfaceFmt, options.MaxWidth);
	CHECK_FAST_ALLOCATION(h_Result.reset(static_cast<unsigned char *>(allocator.allocate(pitch * options.MaxHeight * sizeof(unsigned char)))));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;
	CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hSam, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hHostToDevice, &tmp));
	requestedMemSpace += tmp;
	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t Sam::Transform(std::list<Image<FastAllocator>> &image) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t hostToDeviceTimer = NULL;
	fastGpuTimerHandle_t madTimer = NULL;
	fastGpuTimerHandle_t lutTimer = NULL;
	fastGpuTimerHandle_t deviceToHostTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
		fastGpuTimerCreate(&madTimer);
		fastGpuTimerCreate(&lutTimer);
		fastGpuTimerCreate(&deviceToHostTimer);
	}

	for (auto i = image.begin(); i != image.end(); ++i) {
		Image<FastAllocator> &img = *i;

		printf("Input image: %s\nImage size: %dx%d pixels\n\n", img.inputFileName.c_str(), img.w, img.h);

		if (img.w > maxWidth || img.h > maxHeight) {
			fprintf(stderr, "Unsupported image size\n");
			continue;
		}

		if (info) {
			fastGpuTimerStart(hostToDeviceTimer);
		}

		CHECK_FAST(fastImportFromHostCopy(
			hHostToDevice,

			img.data.get(),
			img.w,
			img.wPitch,
			img.h
		));

		if (info) {
			fastGpuTimerStop(hostToDeviceTimer);
			fastGpuTimerGetTime(hostToDeviceTimer, &elapsedTimeGpu);

			fullTime += elapsedTimeGpu;
			printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);

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

			fullTime += elapsedTimeGpu;
			printf("MAD16 time = %.2f ms\n", elapsedTimeGpu);
		}

		if (hLut != NULL)
		{
			if (info) {
				fastGpuTimerStart(lutTimer);
			}
			CHECK_FAST(fastImageFiltersTransform(
				hLut,
				NULL,

				img.w,
				img.h
			));

			if (info) {
				fastGpuTimerStop(lutTimer);
				fastGpuTimerGetTime(lutTimer, &elapsedTimeGpu);

				fullTime += elapsedTimeGpu;
				printf("LUT time = %.2f ms\n", elapsedTimeGpu);
			}
		}


		if (info) {
			fastGpuTimerStart(deviceToHostTimer);
		}

		fastExportParameters_t exportParameters = { };
		exportParameters.convert = convertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
		CHECK_FAST(fastExportToHostCopy(
			hDeviceToHost,

			h_Result.get(),
			img.w,
			GetPitchFromSurface(surfaceFmt, img.w),
			img.h,

			&exportParameters
		));

		if (info) {
			fastGpuTimerStop(deviceToHostTimer);
			fastGpuTimerGetTime(deviceToHostTimer, &elapsedTimeGpu);

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
		fastGpuTimerDestroy(hostToDeviceTimer);
		fastGpuTimerDestroy(madTimer);
		fastGpuTimerDestroy(lutTimer);
		fastGpuTimerDestroy(deviceToHostTimer);
	}

	return FAST_OK;
}

fastStatus_t Sam::Close(void) const {
	CHECK_FAST(fastImageFiltersDestroy(hSam));
	CHECK_FAST(fastImportFromHostDestroy(hHostToDevice));
	CHECK_FAST(fastExportToHostDestroy(hDeviceToHost));

	return FAST_OK;
}
