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

#include "BaseColorCorrection.h"

#include "supported_files.hpp"
#include "checks.h"

fastStatus_t BaseColorCorrection::Init(BaseColorCorrectionSampleOptions &options) {
	folder = options.IsFolder;
	convertToBGR = options.ConvertToBGR;
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;
	RepeatCount = options.RepeatCount;

	CHECK_FAST( fastImportFromHostCreate(
		&hHostToDeviceAdapter,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	) );

	if ( options.SurfaceFmt == FAST_BGR8 ) {
		options.SurfaceFmt = FAST_RGB8;
	}

	if (!options.BaseColorCorrection.BaseColorCorrectionEnabled)
		return FAST_INVALID_VALUE;

	fastBaseColorCorrection_t baseColorCorrectionParameter = { 0 };
	memcpy(baseColorCorrectionParameter.matrix, options.BaseColorCorrection.BaseColorCorrection, 12 * sizeof(float));
	memcpy(baseColorCorrectionParameter.whiteLevel, options.BaseColorCorrection.WhiteLevel, 3 * sizeof(int));

	CHECK_FAST( fastImageFilterCreate(
		&hBaseColorCorrection,

		FAST_BASE_COLOR_CORRECTION,
		(void *)&baseColorCorrectionParameter,
		
		options.MaxWidth,
		options.MaxHeight,
		
		srcBuffer,
		&baseColorCorrectionBuffer
	) );
	
	CHECK_FAST( fastExportToHostCreate(
		&hDeviceToHostAdapter,

		&surfaceFmt,
		
		baseColorCorrectionBuffer
	) );

	unsigned imageWidth = ((options.MaxWidth * GetBytesPerChannelFromSurface(surfaceFmt) + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT) * FAST_ALIGNMENT;
	unsigned pitch = imageWidth * GetNumberOfChannelsFromSurface(options.SurfaceFmt);
	
	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(buffer.reset((unsigned char *)alloc.allocate(pitch * options.MaxHeight)));

	size_t requestedMemSpace = 0;
	size_t tmp;
	CHECK_FAST( fastImportFromHostGetAllocatedGpuMemorySize( hHostToDeviceAdapter, &tmp ) );
	requestedMemSpace += tmp;
	CHECK_FAST( fastImageFiltersGetAllocatedGpuMemorySize( hBaseColorCorrection, &tmp ) );
	requestedMemSpace += tmp;
	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / ( 1024.0 * 1024.0 ) );

	return FAST_OK;
}

fastStatus_t BaseColorCorrection::Transform(std::list< Image<FastAllocator> > &image) {
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

		if ( img.w > maxWidth ||
			 img.h > maxHeight ) {
				 fprintf(stderr, "Unsupported image size\n");
				 continue;
		}

		if (info) {
			fastGpuTimerStart(hostToDeviceTimer);
		}
		for (unsigned j = 0; j < RepeatCount; j++) {

			CHECK_FAST(fastImportFromHostCopy(
				hHostToDeviceAdapter,

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
				hBaseColorCorrection,
				NULL,

				img.w,
				img.h
			));

			if (info) {
				fastGpuTimerStop(imageFilterTimer);
				fastGpuTimerGetTime(imageFilterTimer, &elapsedTimeGpu);

				totalTime += elapsedTimeGpu;
				printf("Color correction time = %.2f ms\n", elapsedTimeGpu);

				fastGpuTimerStart(deviceToHostTimer);
			}

			fastExportParameters_t exportParameters = { };
			exportParameters.convert = convertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
			CHECK_FAST(fastExportToHostCopy(
				hDeviceToHostAdapter,

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
		}

		printf("Output image: %s\n\n", img.outputFileName.c_str());

		CHECK_FAST_SAVE_FILE( fvSaveImageToFile(
			(char *)img.outputFileName.c_str(),

			buffer,
			surfaceFmt,
			img.h,
			img.w,
			img.wPitch,
			false
		) );
	}

	if (info) {
		printf("Processing time on GPU for %d images images including all transfers = %.2f ms\n", static_cast<unsigned>(RepeatCount * image.size()), totalTime);
		fastGpuTimerDestroy(hostToDeviceTimer);
		fastGpuTimerDestroy(imageFilterTimer);
		fastGpuTimerDestroy(deviceToHostTimer);
	}

	return FAST_OK;
}

fastStatus_t BaseColorCorrection::Close() const {
	CHECK_FAST(fastImageFiltersDestroy(hBaseColorCorrection));
	CHECK_FAST(fastExportToHostDestroy(hDeviceToHostAdapter));
	CHECK_FAST(fastImportFromHostDestroy(hHostToDeviceAdapter));

	return FAST_OK;
}
