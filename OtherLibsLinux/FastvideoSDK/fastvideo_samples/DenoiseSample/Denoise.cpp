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
#include <string.h>
#include <list>

#include "timing.hpp"
#include "helper_image/supported_files.hpp"

#include "fastvideo_denoise.h"
#include "Denoise.h"
#include "EnumToStringDenoise.h"

fastStatus_t Denoise::Init(DenoiseOptions &options) {
	this->options = options;
	folder = options.IsFolder;
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;
	surfaceFmt = options.SurfaceFmt;
	convertToBGR = options.ConvertToBGR;

	parameters.wavelet = options.Wavelet;
	parameters.function = options.Function;

	dwt_levels = options.DWT_Levels;
	for (int i = 0; i < 3; i++) {
		threshold[i] = options.Threshold[i];
		enhance[i] = options.Enhance[i];
	}

	fastSdkParametersHandle_t hSdkParameters;

	CHECK_FAST(fastGetSdkParametersHandle(&hSdkParameters));
	CHECK_FAST(fastDenoiseLibraryInit(hSdkParameters));

	CHECK_FAST(fastImportFromHostCreate(
		&hHostToDeviceAdapter,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	if (options.Info) {
		printf("Wavelet = %s\n", EnumToString(parameters.wavelet));
		printf("Resolution levels = %d\n", dwt_levels + 1);
		printf("Denoising threshold = %g; %g; %g\n", threshold[0], threshold[1], threshold[2]);
		if (enhance[0] != 1.0f || enhance[1] != 1.0f || enhance[2] != 1.0f) {
			printf("Enhancement coefficient = %g; %g; %g\n", enhance[0], enhance[1], enhance[2]);
		}
		printf("Thresholding function = %s\n", EnumToString(parameters.function));
	}

	CHECK_FAST(fastDenoiseCreate(
		&hDenoise,

		options.SurfaceFmt,
		(void *)&parameters,
		options.MaxWidth,
		options.MaxHeight,

		srcBuffer,
		&dstBuffer
	));

	CHECK_FAST(fastExportToHostCreate(
		&hDeviceToHostAdapter,

		&options.SurfaceFmt,

		dstBuffer
	));

	const unsigned pitch = GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth);
	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(buffer.reset((unsigned char *)alloc.allocate(pitch * options.MaxHeight)));

	size_t requestedMemSize = 0;
	size_t componentMemSize = 0;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hHostToDeviceAdapter, &componentMemSize));
	requestedMemSize += componentMemSize;
	CHECK_FAST(fastDenoiseGetAllocatedGpuMemorySize(hDenoise, &componentMemSize));
	requestedMemSize += componentMemSize;
	printf("\nRequested GPU memory size: %.2f MB\n", requestedMemSize / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t Denoise::Transform(std::list< Image<FastAllocator> > &images) {
	fastGpuTimerHandle_t hostToDeviceTimer = NULL;
	fastGpuTimerHandle_t deviceToHostTimer = NULL;
	fastGpuTimerHandle_t denoiseTimer = NULL;

	float elapsedTimeGpu = 0.;
	float totalTime = 0.;

	float processAvgTimeGpu = 0.;
	float processMinTimeGpu = 99999.;
	float processMaxTimeGpu = -99999.;

	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
		fastGpuTimerCreate(&deviceToHostTimer);
	}
	fastGpuTimerCreate(&denoiseTimer);

	double totalFileSize = 0.0;
	for (auto i = images.begin(); i != images.end(); ++i) {
		Image<FastAllocator> &img = *i;

		printf("Input image: %s\n", img.inputFileName.c_str());
		printf("Image size: %dx%d pixels\n", img.w, img.h);
		printf("Image format: %d channel(s), %d bits per channel\n", GetNumberOfChannelsFromSurface(img.surfaceFmt), GetBitsPerChannelFromSurface(surfaceFmt));
	
		if (img.w > maxWidth || img.h > maxHeight) {
			fprintf(stderr, "Image size exceeds the specified maximum size\n");
			continue;
		}

		for (int j = 0; j < options.RepeatCount; j++) {

			if (info) {
				fastGpuTimerStart(hostToDeviceTimer);
			}

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
			}

			denoise_parameters_t dynamic_parameters = { 0 };
			dynamic_parameters.dwt_levels = dwt_levels;
			for (int j = 0; j < 3; j++) {
				dynamic_parameters.threshold[j] = threshold[j];
				dynamic_parameters.enhance[j] = enhance[j];
			}
			for (int j = 0; j < 33; j++)
				dynamic_parameters.threshold_per_level[j] = 1.0f;

			fastGpuTimerStart(denoiseTimer);

			CHECK_FAST(fastDenoiseTransform(
				hDenoise,
				(void *)&dynamic_parameters,

				img.w,
				img.h
			));

			 {
				fastGpuTimerStop(denoiseTimer);
				fastGpuTimerGetTime(denoiseTimer, &elapsedTimeGpu);
				totalTime += elapsedTimeGpu;

				processMaxTimeGpu = std::max(processMaxTimeGpu, elapsedTimeGpu);
				processMinTimeGpu = std::min(processMinTimeGpu, elapsedTimeGpu);
				processAvgTimeGpu += elapsedTimeGpu;
				
			}

			if (info) fastGpuTimerStart(deviceToHostTimer);

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

				totalTime += elapsedTimeGpu;
				printf("Device-to-host transfer = %.2f ms\n", elapsedTimeGpu);
			}
		}


		const double inSize = img.h * img.w;
		totalFileSize += options.RepeatCount * inSize;

		printf("Output image: %s\n\n", img.outputFileName.c_str());

		CHECK_FAST_SAVE_FILE(fvSaveImageToFile(
			(char *)img.outputFileName.c_str(),

			buffer,
			surfaceFmt,
			img.h,
			img.w,
			img.wPitch,

			false // bool info;
		));
	}

	const unsigned imageCount = static_cast<unsigned>(images.size() * options.RepeatCount);
	if (info) {
		printf("Processing time on GPU for %d images including all transfers = %.2f ms; %.0f MPixel/s;  %.0f FPS\n",
			imageCount, totalTime, totalFileSize / (totalTime * 1000.0), 1000.0 * imageCount / totalTime);

		printf("Processing time on GPU for %d images excluding host-to-device and device-to-host transfers: average = %.3f ms (%.0f MPixel/s;  %.0f FPS), min = %.3f ms, max = %.3f ms\n",
			imageCount, processAvgTimeGpu / imageCount, totalFileSize / (processAvgTimeGpu * 1000.0), 1000.0 * imageCount / processAvgTimeGpu,
			processMinTimeGpu, processMaxTimeGpu);

	}
	else
		printf("Processing time on GPU for %d images excluding host-to-device and device-to-host transfers = %.2f ms; %.0f MPixel/s;  %.0f FPS\n",
			imageCount, totalTime, totalFileSize / (totalTime * 1000.0), 1000.0 * imageCount / totalTime);

	if (info) {
		fastGpuTimerDestroy(hostToDeviceTimer);
		fastGpuTimerDestroy(deviceToHostTimer);
	}
	fastGpuTimerDestroy(denoiseTimer);

	return FAST_OK;
}

fastStatus_t Denoise::Close(void) const {
	CHECK_FAST(fastDenoiseDestroy(hDenoise));
	CHECK_FAST(fastExportToHostDestroy(hDeviceToHostAdapter));
	CHECK_FAST(fastImportFromHostDestroy(hHostToDeviceAdapter));

	return FAST_OK;
}
