/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#pragma once

#include <list>
#include <cstdio>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"

#include "Image.h"
#include "Jpeg2JpegSampleOptions.h"
#include "MultiThreadInfo.hpp"

#include "BatchedQueue.h"

#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"
#include "AsyncFileReader.hpp"
#include "AsyncFileWriter.hpp"

#include "helper_jpeg.hpp"
#include "ResizeHelper.h"
#include "SurfaceTraits.hpp"

template <class Reader>
class Jpeg2JpegAsync {
private:
	fastResizerHandle_t hResizer;
	fastJpegDecoderHandle_t hDecoder;
	fastJpegEncoderHandle_t hEncoder;
	fastImageFiltersHandle_t hImageFilterAfter;
	fastImageFiltersHandle_t hImageFilterBefore;
	fastCropHandle_t hCrop;

	fastDeviceSurfaceBufferHandle_t d_decoderBuffer;
	fastDeviceSurfaceBufferHandle_t d_resizerBuffer;
	fastDeviceSurfaceBufferHandle_t d_imageFilterBufferAfter;
	fastDeviceSurfaceBufferHandle_t d_imageFilterBufferBefore;
	fastDeviceSurfaceBufferHandle_t d_cropBuffer;

	Jpeg2JpegSampleOptions options;

	double maxScaleFactor;

	bool info;
	bool mtMode;

public:
	Jpeg2JpegAsync() {
		this->info = false;
		this->mtMode = false;
	};
	~Jpeg2JpegAsync(void) { };

	fastStatus_t Init(
		BaseOptions* baseOptions,
		MtResult* result,
		int threadId,
		void* specialParams)
	{
		this->options = *((Jpeg2JpegSampleOptions*)baseOptions);
		info = baseOptions->Info;
		mtMode = result != nullptr;

		unsigned resizeInputMaxWidth = options.Crop.IsEnabled ? options.Crop.CropWidth : options.MaxWidth;
		unsigned resizeInputMaxHeight = options.Crop.IsEnabled ? options.Crop.CropHeight : options.MaxHeight;
		this->maxScaleFactor = GetResizeMaxScaleFactor(
			resizeInputMaxWidth, resizeInputMaxHeight, options.Resize
		);

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
		}
		else
			printf("Requested GPU memory space: %.2f MB\n\n", requestedMemSpace / megabyte);
		return FAST_OK;
	}


	fastStatus_t Transform(
		Reader* jfifs,
		JpegAsyncFileWriter<ManagedConstFastAllocator<0>>* imgs,
		unsigned threadId, 
		MtResult *result,
		volatile bool* terminate,
		void* specialParams)
	{
		double fullTime = 0.;

		const hostTimer_t decode_timer = hostTimerCreate();

		size_t totalFileSize = 0;
		size_t imageCount = 0;

		double processTimeAll = 0.0;
		double releaseTimeAll = 0.0;
		double allocTimeAll = 0.0;
		double writerTimeAll = 0.0;
		double readerTimeAll = 0.0;

		unsigned outputPitch = GetPitchFromSurface(options.SurfaceFmt, options.Resize.OutputWidth);
		const unsigned outputSize = outputPitch * options.Resize.OutputHeight;
		ManagedConstFastAllocator<0> alloc;
		while (!(*terminate)) {
			hostTimerStart(decode_timer, info);
			auto jfifInBatch = jfifs->ReadNextFileBatch(threadId);
			double getReaderTime = hostTimerEnd(decode_timer, info);

			if (jfifInBatch.IsEmpty())
				break;

			hostTimerStart(decode_timer, info);
			auto jfifOutBatch = imgs->GetNextWriterBatch(threadId);
			double getWriterTime = hostTimerEnd(decode_timer, info);

			jfifOutBatch.SetFilltedItem(jfifInBatch.GetFilledItem());

			double processTime = 0.0;
			double releaseTime = 0.0;
			double allocTime = 0.0;
			for (int i = 0; i < jfifInBatch.GetFilledItem() && !(*terminate); i++) {
				hostTimerStart(decode_timer, info);
				auto jfifIn = jfifInBatch.At(i);
				auto jfifOut = jfifOutBatch.At(i);

				if (options.MaxHeight < jfifIn->info.height || options.MaxWidth < jfifIn->info.width) {
					fprintf(stderr, "No decoder initialized with these parameters\n");
					continue;
				}

				jfifOut->inputFileName = jfifIn->inputFileName;
				jfifOut->outputFileName = jfifIn->outputFileName;

				jfifOut->bytestream.reset(static_cast<unsigned char*>(alloc.allocate(outputSize)));

				memset(&jfifOut->info, 0, sizeof(fastJfifInfo_t));
				{
					jfifOut->info.restartInterval = options.JpegEncoder.RestartInterval;
					jfifOut->info.jpegFmt = options.JpegEncoder.SamplingFmt;
					jfifOut->info.bitsPerChannel = GetBitsPerChannelFromSurface(options.SurfaceFmt);

					jfifOut->info.exifSections = nullptr;
					jfifOut->info.exifSectionsCount = 0;

					jfifOut->info.jpegMode = FAST_JPEG_SEQUENTIAL_DCT;
				}

				allocTime += hostTimerEnd(decode_timer, info);

				hostTimerStart(decode_timer);

				double scaleFactor = double((options.Crop.IsEnabled ? options.Crop.CropWidth : jfifIn->info.width)) / double(options.Resize.OutputWidth);
				unsigned resizedWidth = options.Resize.OutputWidth;
				unsigned resizedHeight = double((options.Crop.IsEnabled ? options.Crop.CropHeight : jfifIn->info.height)) / scaleFactor;

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

				if (fastJpegDecode(hDecoder, jfifIn->GetFastInfo()) != FAST_OK) {
					fprintf(stderr, "JPEG decoding failed (file %s)\n", jfifIn->inputFileName.c_str());
					continue;
				}

				unsigned currentWidth = jfifIn->info.width;
				unsigned currentHeight = jfifIn->info.height;
				if (options.Crop.IsEnabled) {
					if ((options.Crop.CropWidth + options.Crop.CropLeftTopCoordsX) > jfifIn->info.width) {
						fprintf(stderr, "Crop parameters are incorrect: %d + %d > %d\n", options.Crop.CropWidth, options.Crop.CropLeftTopCoordsX, jfifIn->info.width);
						continue;
					}

					if ((options.Crop.CropHeight + options.Crop.CropLeftTopCoordsY) > jfifIn->info.height) {
						fprintf(stderr, "Crop parameters are incorrect: %d + %d > %d\n", options.Crop.CropHeight, options.Crop.CropLeftTopCoordsY, jfifIn->info.height);
						continue;
					}

					if (fastCropTransform(
						hCrop,

						currentWidth, currentHeight,

						options.Crop.CropLeftTopCoordsX,
						options.Crop.CropLeftTopCoordsY,
						options.Crop.CropWidth,
						options.Crop.CropHeight
					) != FAST_OK) {
						fprintf(stderr, "Image cropping failed (file %s)\n", jfifIn->inputFileName.c_str());
						continue;
					}

					currentWidth = options.Crop.CropWidth;
					currentHeight = options.Crop.CropHeight;
				}

				if (options.ImageFilter.SharpBefore != ImageFilterOptions::DisabledSharpConst) {
					fastGaussianFilter_t gaussParameters = { 0 };
					gaussParameters.sigma = options.ImageFilter.SharpBefore;

					if (fastImageFiltersTransform(
						hImageFilterBefore,
						&gaussParameters,

						currentWidth,
						currentHeight
					) != FAST_OK) {
						fprintf(stderr, "Image filter transform (before resizing) failed (file %s)\n", jfifIn->inputFileName.c_str());
						continue;
					}
				}

				if (fastResizerTransform(
					hResizer,
					FAST_LANCZOS,
					currentWidth, currentHeight,
					resizedWidth, &resizedHeight
				) != FAST_OK) {
					fprintf(stderr, "Image resize failed (file %s)\n", jfifIn->inputFileName.c_str());
					continue;
				}
				currentWidth = resizedWidth;
				currentHeight = resizedHeight;

				if (options.ImageFilter.SharpAfter != ImageFilterOptions::DisabledSharpConst) {
					fastGaussianFilter_t gaussParameters = { 0 };
					gaussParameters.sigma = options.ImageFilter.SharpAfter;

					if (fastImageFiltersTransform(
						hImageFilterAfter,
						&gaussParameters,

						currentWidth,
						currentHeight
					) != FAST_OK) {
						fprintf(stderr, "Image filter transform (after resizing) failed (file %s)\n", jfifIn->inputFileName.c_str());
						continue;
					}
				}

				jfifOut->info.width = currentWidth;
				jfifOut->info.height = currentHeight;

				if (fastJpegEncode(hEncoder, options.JpegEncoder.Quality, jfifOut->GetFastInfo()) != FAST_OK) {
					fprintf(stderr, "JPEG encode failed (file %s)\n", jfifIn->inputFileName.c_str());
					continue;
				}

				const unsigned channelCount = GetNumberOfChannelsFromSurface(options.SurfaceFmt);
				totalFileSize += jfifIn->info.width * jfifIn->info.height * channelCount;

				processTime += hostTimerEnd(decode_timer);

				hostTimerStart(decode_timer, info);
				jfifIn->ReleaseBuffer();
				releaseTime += hostTimerEnd(decode_timer, info);
				imageCount++;
			}

			processTimeAll += processTime;
			releaseTimeAll += releaseTime;
			allocTimeAll += allocTime;
			writerTimeAll += getWriterTime;
			readerTimeAll += getReaderTime;
		}

		imgs->WriterFinished(threadId);

		if (mtMode) {
			fullTime = processTimeAll + releaseTimeAll + allocTimeAll + writerTimeAll + readerTimeAll;
			result->totalTime = fullTime;
			result->totalFileSize = totalFileSize;
			result->pipelineHostTime = processTimeAll;
			result->processedItem = imageCount;
			result->readerWaitTime = readerTimeAll;
			result->writerWaitTime = writerTimeAll;
			result->allocationTime = releaseTimeAll + allocTimeAll;
		}

		hostTimerDestroy(decode_timer);

		return FAST_OK;
	}

	fastStatus_t Close()
	{
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
		return FAST_OK;
	}
};

