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
#include "string.h"
#include <list>

#include "HSVLut3D.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "checks.h"

fastStatus_t HsvLut3D::Init(
	LutSampleOptions &options,

	float *lut3D_R,
	float *lut3D_G,
	float *lut3D_B,
	fast_uint3 lut3DSize
) {
	folder = options.IsFolder;
	convertToBGR = options.ConvertToBGR;
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;

	lutSize = lut3DSize;
	
	CHECK_FAST(fastImportFromHostCreate(
		&hImportFromHost,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(lut_H.reset((float *)alloc.allocate(lutSize.x * lutSize.y * lutSize.z * sizeof(float))));
	CHECK_FAST_ALLOCATION(lut_S.reset((float *)alloc.allocate(lutSize.x * lutSize.y * lutSize.z * sizeof(float))));
	CHECK_FAST_ALLOCATION(lut_V.reset((float *)alloc.allocate(lutSize.x * lutSize.y * lutSize.z * sizeof(float))));
	
	fastHsvLut3D_t lutParameters = { 0 };
	{
		memcpy(lut_H.get(), lut3D_R, lutSize.x * lutSize.y * lutSize.z * sizeof(float));
		memcpy(lut_S.get(), lut3D_G, lutSize.x * lutSize.y * lutSize.z * sizeof(float));
		memcpy(lut_V.get(), lut3D_B, lutSize.x * lutSize.y * lutSize.z * sizeof(float));

		lutParameters.LutH = lut_H.get();
		lutParameters.LutS = lut_S.get();
		lutParameters.LutV = lut_V.get();

		lutParameters.dimH = lutSize.x;
		lutParameters.dimS = lutSize.y;
		lutParameters.dimV = lutSize.z;

		lutParameters.operationH = options.Lut.OperationType[0];
		lutParameters.operationS = options.Lut.OperationType[1];
		lutParameters.operationV = options.Lut.OperationType[2];
	}

	CHECK_FAST(fastImageFilterCreate(
		&hLut,

		FAST_HSV_LUT_3D,
		(void *)&lutParameters,

		options.MaxWidth,
		options.MaxHeight,

		srcBuffer,
		&lutBuffer
	));

	CHECK_FAST(fastExportToHostCreate(
		&hExportToHost,

		&dstSurfaceFmt,

		lutBuffer
	));

	CHECK_FAST_ALLOCATION(buffer.reset((unsigned char *)alloc.allocate(GetPitchFromSurface(dstSurfaceFmt, options.MaxWidth) * options.MaxHeight)));

	size_t requestedMemSpace = 0;
	size_t tmp;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hLut, &tmp));
	requestedMemSpace += tmp;
	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t HsvLut3D::Transform(
	std::list<Image<FastAllocator> > &image,

	float *lut3D_R,
	float *lut3D_G,
	float *lut3D_B,
	fast_uint3 lut3DSize
) {
	fastGpuTimerHandle_t hostToDeviceTimer = NULL;
	fastGpuTimerHandle_t deviceToHostTimer = NULL;
	fastGpuTimerHandle_t imageFilterTimer = NULL;

	float elapsedTimeGpu = 0.;
	float totalTime = 0.;

	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
		fastGpuTimerCreate(&deviceToHostTimer);
		fastGpuTimerCreate(&imageFilterTimer);
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

			fastGpuTimerStart(imageFilterTimer);
		}

		CHECK_FAST(fastImageFiltersTransform(
			hLut,
			NULL,

			img.w,
			img.h
		));

		if (info) {
			fastGpuTimerStop(imageFilterTimer);
			fastGpuTimerGetTime(imageFilterTimer, &elapsedTimeGpu);

			totalTime += elapsedTimeGpu;
			printf("LUT time = %.2f ms\n", elapsedTimeGpu);

			fastGpuTimerStart(deviceToHostTimer);
		}

		fastExportParameters_t exportParameters = { };
		exportParameters.convert = convertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
		CHECK_FAST(fastExportToHostCopy(
			hExportToHost,

			buffer.get(),
			img.w,
			img.wPitch,
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
			img.wPitch,
			false
		));
	}

	if (info) {
		printf("Total time for all images = %.2f ms\n", totalTime);
		fastGpuTimerDestroy(hostToDeviceTimer);
		fastGpuTimerDestroy(imageFilterTimer);
		fastGpuTimerDestroy(deviceToHostTimer);
	}

	return FAST_OK;
}

fastStatus_t HsvLut3D::Close(void) const {
	CHECK_FAST(fastImageFiltersDestroy(hLut));
	CHECK_FAST(fastExportToHostDestroy(hExportToHost));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));
	
	return FAST_OK;
}
