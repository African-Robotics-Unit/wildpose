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

#include "Lut16.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "checks.h"

fastStatus_t Lut16::Init(
	LutSampleOptions &options,
	void *lut_R,
	void *lut_G,
	void *lut_B
) {
	folder = options.IsFolder;
	convertToBGR = options.ConvertToBGR;
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;
	filterType = options.Lut.ImageFilter;
	RepeatCount = options.RepeatCount;

	CHECK_FAST(fastImportFromHostCreate(
		&hImportFromHost,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	switch (filterType) {
		case FAST_LUT_12_12:
		{
			fastLut_12_t *lutParameters_ = new fastLut_12_t;
			memcpy(lutParameters_->lut, lut_R, 4096 * sizeof(unsigned short));

			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_12_12,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}

		case FAST_LUT_12_12_C:
		{
			if (lut_G == NULL || lut_B == NULL) {
				fprintf(stderr, "LUTs for G or B component was not set\n");
				return FAST_INVALID_VALUE;
			}

			fastLut_12_C_t *lutParameters_ = new fastLut_12_C_t;
			memcpy(lutParameters_->lut_R, lut_R, 4096 * sizeof(unsigned short));
			memcpy(lutParameters_->lut_G, lut_G, 4096 * sizeof(unsigned short));
			memcpy(lutParameters_->lut_B, lut_B, 4096 * sizeof(unsigned short));

			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_12_12_C,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}

		case FAST_LUT_12_8:
		{
			fastLut_12_8_t *lutParameters_ = new fastLut_12_8_t;
			memcpy(lutParameters_->lut, lut_R, 4096 * sizeof(unsigned char));

			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_12_8,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}

		case FAST_LUT_12_8_C:
		{
			if (lut_G == NULL || lut_B == NULL) {
				fprintf(stderr, "LUTs for G or B component was not set\n");
				return FAST_INVALID_VALUE;
			}

			fastLut_12_8_C_t *lutParameters_ = new fastLut_12_8_C_t;
			memcpy(lutParameters_->lut_R, lut_R, 4096 * sizeof(unsigned char));
			memcpy(lutParameters_->lut_G, lut_G, 4096 * sizeof(unsigned char));
			memcpy(lutParameters_->lut_B, lut_B, 4096 * sizeof(unsigned char));

			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_12_8_C,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}

		case FAST_LUT_12_16:
		{
			fastLut_12_t *lutParameters_ = new fastLut_12_t;
			memcpy(lutParameters_->lut, lut_R, 4096 * sizeof(unsigned short));

			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_12_16,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}

		case FAST_LUT_12_16_C:
		{
			if (lut_G == NULL || lut_B == NULL) {
				fprintf(stderr, "LUTs for G or B component was not set\n");
				return FAST_INVALID_VALUE;
			}

			fastLut_12_C_t *lutParameters_ = new fastLut_12_C_t;
			memcpy(lutParameters_->lut_R, lut_R, 4096 * sizeof(unsigned short));
			memcpy(lutParameters_->lut_G, lut_G, 4096 * sizeof(unsigned short));
			memcpy(lutParameters_->lut_B, lut_B, 4096 * sizeof(unsigned short));

			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_12_16_C,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}

		case FAST_LUT_16_16:
		{
			fastLut_16_t *lutParameters_ = new fastLut_16_t;
			memcpy(lutParameters_->lut, lut_R, 16384 * sizeof(unsigned short));

			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_16_16,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}

		case FAST_LUT_16_16_C:
		{
			if (lut_G == NULL || lut_B == NULL) {
				fprintf(stderr, "LUTs for G or B component was not set\n");
				return FAST_INVALID_VALUE;
			}

			fastLut_16_C_t *lutParameters_ = new fastLut_16_C_t;
			memcpy(lutParameters_->lut_R, lut_R, 16384 * sizeof(unsigned short));
			memcpy(lutParameters_->lut_G, lut_G, 16384 * sizeof(unsigned short));
			memcpy(lutParameters_->lut_B, lut_B, 16384 * sizeof(unsigned short));

			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_16_16_C,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}

		case FAST_LUT_16_16_FR:
		{
			fastLut_16_FR_t *lutParameters_ = new fastLut_16_FR_t;
			memcpy(lutParameters_->lut, lut_R, 65536 * sizeof(unsigned short));

			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_16_16_FR,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}

		case FAST_LUT_16_16_FR_C:
		{
			if (lut_G == NULL || lut_B == NULL) {
				fprintf(stderr, "LUTs for G or B component was not set\n");
				return FAST_INVALID_VALUE;
			}

			fastLut_16_FR_C_t *lutParameters_ = new fastLut_16_FR_C_t;
			memcpy(lutParameters_->lut_R, lut_R, 65536 * sizeof(unsigned short));
			memcpy(lutParameters_->lut_G, lut_G, 65536 * sizeof(unsigned short));
			memcpy(lutParameters_->lut_B, lut_B, 65536 * sizeof(unsigned short));
			
			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_16_16_FR_C,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}

		case FAST_LUT_16_8:
		{
			fastLut_16_8_t *lutParameters_ = new fastLut_16_8_t;
			memcpy(lutParameters_->lut, lut_R, 16384 * sizeof(unsigned char));

			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_16_8,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}

		case FAST_LUT_16_8_C:
		{
			if (lut_G == NULL || lut_B == NULL) {
				fprintf(stderr, "LUTs for G or B component was not set\n");
				return FAST_INVALID_VALUE;
			}

			fastLut_16_8_C_t *lutParameters_ = new fastLut_16_8_C_t;
			memcpy(lutParameters_->lut_R, lut_R, 16384 * sizeof(unsigned char));
			memcpy(lutParameters_->lut_G, lut_G, 16384 * sizeof(unsigned char));
			memcpy(lutParameters_->lut_B, lut_B, 16384 * sizeof(unsigned char));

			CHECK_FAST(fastImageFilterCreate(
				&hLut,

				FAST_LUT_16_8_C,
				lutParameters_,

				options.MaxWidth,
				options.MaxHeight,

				srcBuffer,
				&lutBuffer
			));
			lutParameters = lutParameters_;

			break;
		}
	}
	
	CHECK_FAST(fastExportToHostCreate(
		&hExportToHost,

		&dstSurfaceFmt,

		lutBuffer
	));

	FastAllocator alloc;
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

fastStatus_t Lut16::Transform(
	std::list< Image<FastAllocator> > &image,

	void *lut_R,
	void *lut_G,
	void *lut_B
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

		if ( img.w > maxWidth ||
			 img.h > maxHeight ) {
				 fprintf(stderr, "Unsupported image size\n");
				 continue;
		}

		if (info) {
			fastGpuTimerStart(hostToDeviceTimer);
		}
		unsigned outputPitch = GetPitchFromSurface(dstSurfaceFmt, img.w);

		for (unsigned j = 0; j < RepeatCount; j++) {
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

			if (info) {
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
				outputPitch,
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
			dstSurfaceFmt,
			img.h,
			img.w,
			outputPitch,
			false
		) );
	}

	if (info) {
		printf("Processing time on GPU for %d images images including all transfers = %.2f ms\n", static_cast<unsigned>(RepeatCount * image.size()),  totalTime);
		fastGpuTimerDestroy(hostToDeviceTimer);
		fastGpuTimerDestroy(imageFilterTimer);
		fastGpuTimerDestroy(deviceToHostTimer);
	}

	return FAST_OK;
}

fastStatus_t Lut16::Close() const {
	CHECK_FAST(fastImageFiltersDestroy(hLut));
	CHECK_FAST(fastExportToHostDestroy(hExportToHost));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));

	if (lutParameters != NULL) {
		delete lutParameters;
	}
	
	return FAST_OK;
}
