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

#include "Resize.h"
#include "checks.h"
#include "supported_files.hpp"

#include "fastvideo_sdk.h"

fastStatus_t Resizer::Init(ResizerSampleOptions &options, const double maxScaleFactor) {
	folder = options.IsFolder;
	convertToBGR = options.ConvertToBGR;
	this->maxScaleFactor = maxScaleFactor;
	this->options = options;

	CHECK_FAST(fastImportFromHostCreate(
		&hHostToDeviceAdapter,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	CHECK_FAST(fastResizerCreate(
		&hResizer,

		options.MaxWidth,
		options.MaxHeight,

		options.Resize.OutputWidth,
		options.Resize.OutputHeight,

		maxScaleFactor,
		
		options.Resize.ShiftX,
		options.Resize.ShiftY,

		srcBuffer,
		&dstBuffer
	));

	CHECK_FAST(fastExportToHostCreate(
		&hDeviceToHostAdapter,

		&options.SurfaceFmt,

		dstBuffer
	));

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(byteStream.reset((unsigned char *)alloc.allocate(
		GetPitchFromSurface(options.SurfaceFmt, options.Resize.OutputWidth) * options.Resize.OutputHeight)
	));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;

	CHECK_FAST(fastResizerGetAllocatedGpuMemorySize(hResizer, &tmp));
	requestedMemSpace += tmp;

	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hHostToDeviceAdapter, &tmp));
	requestedMemSpace += tmp;

	printf("Requested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t Resizer::Close() const {
	CHECK_FAST(fastResizerDestroy(hResizer));
	CHECK_FAST(fastImportFromHostDestroy(hHostToDeviceAdapter));
	CHECK_FAST(fastExportToHostDestroy(hDeviceToHostAdapter));

	return FAST_OK;
}

fastStatus_t Resizer::Resize(std::list< Image<FastAllocator> > &inputImages, const char *outputPattern) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t hostToDeviceTimer = NULL;
	fastGpuTimerHandle_t resizeTimer = NULL;
	fastGpuTimerHandle_t deviceToHostTimer = NULL;

	float processAvgTimeGpu = 0.;
	float processMinTimeGpu = 99999.;
	float processMaxTimeGpu = -99999.;

	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
		fastGpuTimerCreate(&deviceToHostTimer);
	}
	fastGpuTimerCreate(&resizeTimer);

	double totalFileSize = 0.0;
	for (auto i = inputImages.begin(); i != inputImages.end(); ++i) {
		const unsigned width = (*i).w;
		const unsigned height = (*i).h;

		if (options.MaxWidth < width ||
			options.MaxHeight < height) {
			fprintf(stderr, "No decoder initialized with these parameters\n");
			continue;
		}

		printf("Input image: %s\nInput image size: %dx%d pixels\n", (*i).inputFileName.c_str(), width, height);

		double scaleFactorX = double(width) / double(options.Resize.OutputWidth);
		double scaleFactor = scaleFactorX;

		if (options.Resize.OutputHeightEnabled)
		{
			double scaleFactorY = double(height) / double(options.Resize.OutputHeight);
			scaleFactor = scaleFactorX > scaleFactorY ? scaleFactorX : scaleFactorY;
		}
		unsigned resizedHeight = 0;

		if (scaleFactor > maxScaleFactor) {
			fprintf(stderr, "Image scale factor (%.3f) is more than maxScaleFactor (%.3f)\n\n", scaleFactor, maxScaleFactor);
			continue;
		}

		if (options.Resize.OutputWidth < FAST_MIN_SCALED_SIZE) {
			fprintf(stderr, "Image width %d is not supported (the smallest image size is %dx%d)\n", options.Resize.OutputWidth, FAST_MIN_SCALED_SIZE, FAST_MIN_SCALED_SIZE);
			continue;
		}

		if (options.Resize.OutputHeightEnabled && options.Resize.OutputHeight < FAST_MIN_SCALED_SIZE) {
			fprintf(stderr, "Image height %d is not supported (the smallest image size is %dx%d)\n", options.Resize.OutputHeight, FAST_MIN_SCALED_SIZE, FAST_MIN_SCALED_SIZE);
			continue;
		}

		printf("Image scale factor: %.3f\n", scaleFactor);


		for (int j = 0; j < options.RepeatCount; j++) {
			if (info) {
				fastGpuTimerStart(hostToDeviceTimer);
			}

			CHECK_FAST(fastImportFromHostCopy(
				hHostToDeviceAdapter,

				(*i).data.get(),
				(*i).w,
				(*i).wPitch,
				(*i).h
			));

			if (info) {
				fastGpuTimerStop(hostToDeviceTimer);
				fastGpuTimerGetTime(hostToDeviceTimer, &elapsedTimeGpu);

				fullTime += elapsedTimeGpu;
				printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);

			}

			fastGpuTimerStart(resizeTimer);

			if (options.Resize.BackgroundEnabled) {
				fastRgb_t background = { 0 };
				{
					background.R = options.Resize.Background[0];
					background.G = options.Resize.Background[1];
					background.B = options.Resize.Background[2];
				};
				CHECK_FAST(fastResizerTransformWithPaddingCentered(
					hResizer,
					FAST_LANCZOS,
					width, height,
					background,
					options.Resize.OutputWidth, options.Resize.OutputHeight
				));
				resizedHeight = options.Resize.OutputHeight;
			} else {
				if (options.Resize.OutputHeightEnabled) {
					CHECK_FAST(fastResizerTransformStretch(
						hResizer, FAST_LANCZOS, width, height, options.Resize.OutputWidth, options.Resize.OutputHeight
					));
					resizedHeight = options.Resize.OutputHeight;
				} else {
					CHECK_FAST(fastResizerTransform(
						hResizer, FAST_LANCZOS, width, height, options.Resize.OutputWidth, &resizedHeight
					));
				}
			}
			
			{
				fastGpuTimerStop(resizeTimer);
				fastGpuTimerGetTime(resizeTimer, &elapsedTimeGpu);
				fullTime += elapsedTimeGpu;

				processMaxTimeGpu = std::max(processMaxTimeGpu, elapsedTimeGpu);
				processMinTimeGpu = std::min(processMinTimeGpu, elapsedTimeGpu);
				processAvgTimeGpu += elapsedTimeGpu;
			}

			if (info) fastGpuTimerStart(deviceToHostTimer);

			fastExportParameters_t exportParameters = { };
			exportParameters.convert = convertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
			CHECK_FAST(fastExportToHostCopy(
				hDeviceToHostAdapter,

				byteStream.get(),
				options.Resize.OutputWidth,
				GetPitchFromSurface(options.SurfaceFmt, options.Resize.OutputWidth),
				resizedHeight,

				&exportParameters
			));

			if (info) {
				fastGpuTimerStop(deviceToHostTimer);
				fastGpuTimerGetTime(deviceToHostTimer, &elapsedTimeGpu);

				fullTime += elapsedTimeGpu;
				printf("Device-to-host transfer = %.2f ms\n\n", elapsedTimeGpu);
			}
		}

		printf("Result configuration: %dx%d\n", options.Resize.OutputWidth, resizedHeight);

		CHECK_FAST(fvSaveImageToFile(
			(char *)(*i).outputFileName.c_str(),
			byteStream,
			(*i).surfaceFmt,
			resizedHeight,
			options.Resize.OutputWidth,
			GetPitchFromSurface(options.SurfaceFmt, options.Resize.OutputWidth),
			false
		));

		printf("Output image: %s\nOutput image size: %dx%d pixels\n\n", (*i).outputFileName.c_str(), options.Resize.OutputWidth, resizedHeight);


		const double inSize = (*i).h * (*i).w;
		totalFileSize += inSize * options.RepeatCount;
	}

	const unsigned imageCount = static_cast<unsigned>(inputImages.size() * options.RepeatCount);
	
	if (info) {
		printf("Processing time on GPU for %d images including all transfers = %.2f ms; %.0f MPixel/s;  %.0f FPS\n",
			imageCount, fullTime, totalFileSize / (fullTime * 1000.0), 1000.0 * imageCount / fullTime);

		printf("Processing time on GPU for %d images excluding host-to-device and device-to-host transfers: average = %.3f ms (%.0f MPixel/s;  %.0f FPS), min = %.3f ms, max = %.3f ms\n",
			imageCount, processAvgTimeGpu / imageCount, totalFileSize / (processAvgTimeGpu * 1000.0), 1000.0 * imageCount / processAvgTimeGpu,
			processMinTimeGpu, processMaxTimeGpu);

	}
	else
		printf("Processing time on GPU for %d images excluding host-to-device and device-to-host transfers = %.2f ms; %.0f MPixel/s;  %.0f FPS\n",
			imageCount, fullTime, totalFileSize / (fullTime * 1000.0), 1000.0 * imageCount / fullTime);


	if (info) {
		fastGpuTimerDestroy(hostToDeviceTimer);
		fastGpuTimerDestroy(deviceToHostTimer);
	}
	fastGpuTimerDestroy(resizeTimer);

	return FAST_OK;
}
