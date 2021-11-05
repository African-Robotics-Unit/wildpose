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
#include <math.h>

#ifndef __GNUC__
#include <Windows.h>
#endif

#include "Jpeg2Jpeg.h"
#include "timing.hpp"
#include "checks.h"
#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "helper_jpeg.hpp"

#include "ResizeHelper.h"


fastStatus_t Jpeg2Jpeg::Init(Jpeg2JpegSampleOptions &options, double maxScaleFactor, MtResult *result) {
	this->options = options;
	this->maxScaleFactor = maxScaleFactor;
	channelCount = options.SurfaceFmt == FAST_I8 ? 1 : 3;
	
	unsigned currentMaxWidth = options.MaxWidth;
	unsigned currentMaxHeight = options.MaxHeight;

	CHECK_FAST(fastJpegDecoderCreate(
		&hDecoder,

		options.SurfaceFmt,
		currentMaxWidth,
		currentMaxHeight,
		true,
		&d_decoderBuffer
	));
	fastDeviceSurfaceBufferHandle_t buffer = d_decoderBuffer;

	if (options.Crop.IsEnabled) {
		CHECK_FAST(fastCropCreate(
			&hCrop,

			currentMaxWidth,
			currentMaxHeight,

			options.Crop.CropWidth,
			options.Crop.CropHeight,

			buffer,
			&d_cropBuffer
		));
		buffer = d_cropBuffer;
		currentMaxWidth = options.Crop.CropWidth;
		currentMaxHeight = options.Crop.CropHeight;
	}

	if (options.ImageFilter.SharpBefore != ImageFilterOptions::DisabledSharpConst) {
		CHECK_FAST(fastImageFilterCreate(
			&hImageFilterBefore,

			FAST_GAUSSIAN_SHARPEN,
			NULL,

			currentMaxWidth,
			currentMaxHeight,
			buffer,
			&d_imageFilterBufferBefore
		));
		buffer = d_imageFilterBufferBefore;
	}
		
	unsigned outputPitch = GetPitchFromSurface(options.SurfaceFmt, options.Resize.OutputWidth);
	
	CHECK_FAST(fastResizerCreate(
		&hResizer,

		currentMaxWidth,
		currentMaxHeight,

		options.Resize.OutputWidth,
		options.Resize.OutputHeight,

		maxScaleFactor,

		options.Resize.ShiftX,
		options.Resize.ShiftY,

		buffer,
		&d_resizerBuffer
	));
	buffer = d_resizerBuffer;

	currentMaxWidth = options.Resize.OutputWidth;
	currentMaxHeight = options.Resize.OutputHeight;

	if (options.ImageFilter.SharpAfter != ImageFilterOptions::DisabledSharpConst) {
		CHECK_FAST(fastImageFilterCreate(
			&hImageFilterAfter,

			FAST_GAUSSIAN_SHARPEN,
			NULL,

			currentMaxWidth,
			currentMaxHeight,

			buffer,
			&d_imageFilterBufferAfter
		));
		buffer = d_imageFilterBufferAfter;
	}

	FastAllocator alloc;
	h_ResizedJpegStream.reset((unsigned char *)alloc.allocate(outputPitch * options.Resize.OutputWidth));

	CHECK_FAST(fastJpegEncoderCreate(
		&hEncoder,
		currentMaxWidth,
		currentMaxHeight,
		buffer
	));

	size_t requestedMemSpace = 0;
	size_t tmp;
	CHECK_FAST(fastJpegDecoderGetAllocatedGpuMemorySize(hDecoder, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastJpegEncoderGetAllocatedGpuMemorySize(hEncoder, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastResizerGetAllocatedGpuMemorySize(hResizer, &tmp));
	requestedMemSpace += tmp;
	if (options.ImageFilter.SharpAfter != ImageFilterOptions::DisabledSharpConst) {
		CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hImageFilterAfter, &tmp));
		requestedMemSpace += tmp;
	}
	if (options.ImageFilter.SharpBefore != ImageFilterOptions::DisabledSharpConst) {
		CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hImageFilterBefore, &tmp));
		requestedMemSpace += tmp;
	}
	if (options.Crop.IsEnabled) {
		CHECK_FAST(fastCropGetAllocatedGpuMemorySize(hCrop, &tmp));
		requestedMemSpace += tmp;
	}
	const double megabyte = 1024.0 * 1024.0;
	if (mtMode && result != nullptr) {
		result->requestedMemSize = requestedMemSpace / megabyte;
	} else {
		printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / megabyte);
	}

	maxBufferSize = jfifInfo.bytestreamSize = outJfifInfo.bytestreamSize = channelCount * options.MaxHeight * options.MaxWidth;
	CHECK_FAST(fastMalloc((void **)&jfifInfo.h_Bytestream, jfifInfo.bytestreamSize));
	CHECK_FAST(fastMalloc((void **)&outJfifInfo.h_Bytestream, outJfifInfo.bytestreamSize));

	outJfifInfo.exifSectionsCount = 0;
	jfifInfo.exifSections = NULL;

	return FAST_OK;
}

fastStatus_t Jpeg2Jpeg::Close() const {
	CHECK_FAST(fastJpegDecoderDestroy(hDecoder));
	CHECK_FAST(fastJpegEncoderDestroy(hEncoder));
	CHECK_FAST(fastResizerDestroy(hResizer));
	if (options.ImageFilter.SharpBefore != ImageFilterOptions::DisabledSharpConst) {
		CHECK_FAST(fastImageFiltersDestroy(hImageFilterBefore));
	}
	if (options.ImageFilter.SharpAfter != ImageFilterOptions::DisabledSharpConst) {
		CHECK_FAST(fastImageFiltersDestroy(hImageFilterAfter));
	}

	if (options.Crop.IsEnabled) {
		CHECK_FAST(fastCropDestroy(hCrop));
	}

	if (jfifInfo.h_Bytestream != NULL)
		CHECK_FAST_DEALLOCATION(fastFree(jfifInfo.h_Bytestream));

	if (outJfifInfo.h_Bytestream != NULL)
		CHECK_FAST_DEALLOCATION(fastFree(outJfifInfo.h_Bytestream));

	for (unsigned i = 0; i < jfifInfo.exifSectionsCount; i++) {
		free(jfifInfo.exifSections[i].exifData);
	}

	if (jfifInfo.exifSections != NULL) {
		free(jfifInfo.exifSections);
	}

	return FAST_OK;
}

fastStatus_t Jpeg2Jpeg::Resize(std::list< Bytestream<FastAllocator> > &inputImages, int threadId, MtResult *result) {
	float totalTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t decoderTimer = NULL;
	fastGpuTimerHandle_t resizeTimer = NULL;
	fastGpuTimerHandle_t encoderTimer = NULL;
	fastGpuTimerHandle_t imageFilterTimerBefore = NULL;
	fastGpuTimerHandle_t imageFilterTimerAfter = NULL;
	fastGpuTimerHandle_t cropTimer = NULL;

	hostTimer_t timer = hostTimerCreate();

	CHECK_FAST(fastGpuTimerCreate(&decoderTimer));
	CHECK_FAST(fastGpuTimerCreate(&resizeTimer));
	CHECK_FAST(fastGpuTimerCreate(&encoderTimer));

	if (info) {
		CHECK_FAST(fastGpuTimerCreate(&imageFilterTimerAfter));
		CHECK_FAST(fastGpuTimerCreate(&imageFilterTimerBefore));
		CHECK_FAST(fastGpuTimerCreate(&cropTimer));
	}

#ifndef __GNUC__
	if (options.MultiProcess) {
		printf("Waiting\n");
		HANDLE hEvent = OpenEvent(EVENT_ALL_ACCESS, TRUE, TEXT("Global/Start"));
		WaitForSingleObject(hEvent, INFINITE);
	}
#endif

	double totalFileSize = 0.0;
	for (auto i = inputImages.begin(); i != inputImages.end(); ++i) {
		jfifInfo.bytestreamSize = maxBufferSize;

		if (fastJfifLoadFromMemory((*i).data.get(), (*i).size, &jfifInfo) != FAST_OK) {
			fprintf(stderr, "Loading from memory failed (file %s)\n", (*i).inputFileName.c_str());
			continue;
		}
		
		const double inSize = jfifInfo.width * jfifInfo.height;
		totalFileSize += options.RepeatCount * inSize;

		if (threadId == 0) {
			printf("Input image: %s\nInput image size: %dx%d pixels\n", (*i).inputFileName.c_str(), jfifInfo.width, jfifInfo.height);
			printf("Input sampling format: %s\n", EnumToString(jfifInfo.jpegFmt));
			printf("Input restart interval: %d\n\n", jfifInfo.restartInterval);
		}

		if (options.MaxHeight < jfifInfo.height || options.MaxWidth < jfifInfo.width) {
			fprintf(stderr, "No decoder initialized with these parameters\n");
			continue;
		}

		double scaleFactor = double((options.Crop.IsEnabled ? options.Crop.CropWidth : jfifInfo.width)) / double(options.Resize.OutputWidth);
		unsigned resizedWidth = options.Resize.OutputWidth;
		unsigned resizedHeight = double((options.Crop.IsEnabled ? options.Crop.CropHeight : jfifInfo.height)) / scaleFactor;

		
		if (threadId == 0) {
			printf("Output image: %s\nOutput image size: %dx%d pixels\n\n", (*i).outputFileName.c_str(), resizedWidth, static_cast<int>(jfifInfo.height/ scaleFactor));
			printf("Image scale factor: %.3f\n\n", scaleFactor);
		}

		if (scaleFactor > maxScaleFactor) {
			fprintf(stderr, "Image scale factor (%.3f) is more than maxScaleFactor (%.3f)\n\n", scaleFactor, maxScaleFactor);
			continue;
		}

		if (scaleFactor > ResizerOptions::SCALE_FACTOR_MAX) {
			fprintf(stderr, "Incorrect image scale factor (%.3f). Max scale factor is %d\n", scaleFactor, ResizerOptions::SCALE_FACTOR_MAX);
			continue;
		}
	
		if (resizedWidth < FAST_MIN_SCALED_SIZE) {
			fprintf(stderr, "Image width %d is not supported (the smallest image size is %dx%d)\n", resizedWidth, FAST_MIN_SCALED_SIZE, FAST_MIN_SCALED_SIZE);
			continue;
		}

		if (resizedHeight < FAST_MIN_SCALED_SIZE) {
			fprintf(stderr, "Image height %d is not supported (the smallest image size is %dx%d)\n", resizedHeight, FAST_MIN_SCALED_SIZE, FAST_MIN_SCALED_SIZE);
			continue;
		}

		hostTimerStart(timer);
		for (int j = 0; j < options.RepeatCount; j++) {
			unsigned currentWidth, currentHeight;

			if (!mtMode) {
				fastGpuTimerStart(decoderTimer);
			}

			if (fastJpegDecode(hDecoder, &jfifInfo) != FAST_OK) {
				fprintf(stderr, "JPEG decoding failed (file %s)\n", (*i).inputFileName.c_str());
				continue;
			}
			currentWidth = jfifInfo.width;
			currentHeight = jfifInfo.height;

			if (!mtMode) {
				fastGpuTimerStop(decoderTimer);
				fastGpuTimerGetTime(decoderTimer, &elapsedTimeGpu);

				totalTime += elapsedTimeGpu;
				if (info) {
					printf("\nDecode time (includes host-to-device transfer) = %.2f ms\n", elapsedTimeGpu);
				}
			}

			if (options.Crop.IsEnabled) {
				if ((options.Crop.CropWidth + options.Crop.CropLeftTopCoordsX) > jfifInfo.width) {
					fprintf(stderr, "Crop parameters are incorrect: %d + %d > %d\n", options.Crop.CropWidth, options.Crop.CropLeftTopCoordsX, jfifInfo.width);
					continue;
				}

				if ((options.Crop.CropHeight + options.Crop.CropLeftTopCoordsY) > jfifInfo.height) {
					fprintf(stderr, "Crop parameters are incorrect: %d + %d > %d\n", options.Crop.CropHeight, options.Crop.CropLeftTopCoordsY, jfifInfo.height);
					continue;
				}

				if (info) {
					fastGpuTimerStart(cropTimer);
				}

				if (fastCropTransform(
					hCrop,

					currentWidth, currentHeight,

					options.Crop.CropLeftTopCoordsX,
					options.Crop.CropLeftTopCoordsY,
					options.Crop.CropWidth,
					options.Crop.CropHeight
				) != FAST_OK) {
					fprintf(stderr, "Image cropping failed (file %s)\n", (*i).inputFileName.c_str());
					continue;
				}

				currentWidth = options.Crop.CropWidth;
				currentHeight = options.Crop.CropHeight;

				if (info) {
					fastGpuTimerStop(cropTimer);
					fastGpuTimerGetTime(cropTimer, &elapsedTimeGpu);

					totalTime += elapsedTimeGpu;
					if (!mtMode) {
						printf("Crop time = %.2f ms\n", elapsedTimeGpu);
					}
				}
			}

			if (options.ImageFilter.SharpBefore != ImageFilterOptions::DisabledSharpConst) {
				fastGaussianFilter_t gaussParameters = { 0 };
				gaussParameters.sigma = options.ImageFilter.SharpBefore;

				if (info) {
					fastGpuTimerStart(imageFilterTimerBefore);
				}

				if (fastImageFiltersTransform(
					hImageFilterBefore,
					&gaussParameters,

					currentWidth,
					currentHeight
				) != FAST_OK) {
					fprintf(stderr, "Image filter transform (before resizing) failed (file %s)\n", (*i).inputFileName.c_str());
					continue;
				}

				if (info) {
					fastGpuTimerStop(imageFilterTimerBefore);
					fastGpuTimerGetTime(imageFilterTimerBefore, &elapsedTimeGpu);

					totalTime += elapsedTimeGpu;
					if (!mtMode) {
						printf("Sharpen filter (before resize) time = %.2f ms\n", elapsedTimeGpu);
					}
				}
			}

			if (!mtMode) {
				fastGpuTimerStart(resizeTimer);
			}

			if (fastResizerTransform(
				hResizer, 
				FAST_LANCZOS, 
				currentWidth, 
				currentHeight, 
				resizedWidth, 
				&resizedHeight) != FAST_OK) {
				fprintf(stderr, "Image resize failed (file %s)\n", (*i).inputFileName.c_str());
				continue;
			}

			currentWidth = resizedWidth;
			currentHeight = resizedHeight;

			if (!mtMode) {
				fastGpuTimerStop(resizeTimer);
				fastGpuTimerGetTime(resizeTimer, &elapsedTimeGpu);

				totalTime += elapsedTimeGpu;
				if (info)
					printf("Resize time = %.3f ms\n", elapsedTimeGpu);
			}

			if (options.ImageFilter.SharpAfter != ImageFilterOptions::DisabledSharpConst) {
				fastGaussianFilter_t gaussParameters = { 0 };
				gaussParameters.sigma = options.ImageFilter.SharpAfter;

				if (info) {
					fastGpuTimerStart(imageFilterTimerAfter);
				}

				if (fastImageFiltersTransform(
					hImageFilterAfter,
					&gaussParameters,

					currentWidth,
					currentHeight
				) != FAST_OK) {
					fprintf(stderr, "Image filter transform (after resizing) failed (file %s)\n", (*i).inputFileName.c_str());
					continue;
				}

				if (info) {
					fastGpuTimerStop(imageFilterTimerAfter);
					fastGpuTimerGetTime(imageFilterTimerAfter, &elapsedTimeGpu);

					totalTime += elapsedTimeGpu;
					if (!mtMode) {
						printf("Sharpen filter (after resize) time = %.2f ms\n", elapsedTimeGpu);
					}
				}
			}

			outJfifInfo.restartInterval = options.JpegEncoder.RestartInterval;
			outJfifInfo.jpegFmt = options.JpegEncoder.SamplingFmt;
			outJfifInfo.width = currentWidth;
			outJfifInfo.height = currentHeight;
			outJfifInfo.jpegMode = FAST_JPEG_SEQUENTIAL_DCT;

			if (!mtMode) {
				fastGpuTimerStart(encoderTimer);
			}

			if (fastJpegEncode(hEncoder, options.JpegEncoder.Quality, &outJfifInfo) != FAST_OK) {
				fprintf(stderr, "JPEG encode failed (file %s)\n", (*i).inputFileName.c_str());
				continue;
			}

			if (!mtMode) {
				fastGpuTimerStop(encoderTimer);
				fastGpuTimerGetTime(encoderTimer, &elapsedTimeGpu);

				totalTime += elapsedTimeGpu;
				if (info)
					printf("Encode time (includes device-to-host transfer) = %.2f ms\n", elapsedTimeGpu);
			}
		}

		if (mtMode) {
			totalTime += static_cast<float>(hostTimerEnd(timer) * 1000.0f);
		}

		if (fastJfifStoreToFile((*i).outputFileName.c_str(), &outJfifInfo) != FAST_OK) {
			fprintf(stderr, "Store to file failed (file %s)\n", (*i).outputFileName.c_str());
			continue;
		}
	}

	const unsigned imageCount = static_cast<unsigned>(inputImages.size() * options.RepeatCount);

	if (mtMode) {
		result->totalTime = totalTime;
		result->totalFileSize = totalFileSize;
	} else {
		if (info) {
			printf("Processing time on GPU for %d images including all transfers = %.2f ms; %.0f FPS; average = %.2f ms\n\n",
				imageCount, totalTime, imageCount * 1000.0 / totalTime, totalTime / imageCount);
			//printf("Processing time on GPU for %d images excluding host-to-device and device-to-host transfers: average = %.3f ms (%.0f MPixel/s;  %.0f FPS), min = %.3f ms, max = %.3f ms\n",
			//	imageCount, processAvgTimeGpu / imageCount, totalFileSize / (processAvgTimeGpu * 1000.0), 1000.0 * imageCount / processAvgTimeGpu,
			//	processMinTimeGpu, processMaxTimeGpu);

		}
		else
			printf("Processing time on GPU for %d images including host-to-device and device-to-host transfers = %.2f ms; %.0f FPS; average = %.2f ms\n",
				imageCount, totalTime, imageCount * 1000.0 / totalTime, totalTime / imageCount);
	}
	if (threadId == 0) {
		printf("For all output images:\n\tOutput sampling format: %s\n", EnumToString(options.JpegEncoder.SamplingFmt));
		printf("\tJPEG quality: %d%%\n", options.JpegEncoder.Quality);
		printf("\tOutput restart interval: %d\n\n", options.JpegEncoder.RestartInterval);
	}

	if (info) {
		CHECK_FAST(fastGpuTimerDestroy(imageFilterTimerAfter));
		CHECK_FAST(fastGpuTimerDestroy(imageFilterTimerBefore));
		CHECK_FAST(fastGpuTimerDestroy(cropTimer));
	}

	hostTimerDestroy(timer);

	CHECK_FAST(fastGpuTimerDestroy(decoderTimer));
	CHECK_FAST(fastGpuTimerDestroy(resizeTimer));
	CHECK_FAST(fastGpuTimerDestroy(encoderTimer));

	return FAST_OK;
}
