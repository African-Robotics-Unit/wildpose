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
#include "fastvideo_encoder_j2k.h"

#include "Image.h"
#include "J2kEncoderOptions.h"
#include "J2kEncoderBase.h"
#include "MultiThreadInfo.hpp"

#include "BatchedQueue.h"

#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"
#include "AsyncFileReader.hpp"
#include "AsyncFileWriter.hpp"

#include <math.h>
#include "SurfaceTraits.hpp"

template <class Reader>
class J2kEncoderAsync : public J2kEncoderBase {
private:
	fastEncoderJ2kHandle_t hEncoder;
	fastImportFromHostHandle_t hHostToDeviceAdapter;

	fastDeviceSurfaceBufferHandle_t srcBuffer;

	bool info;
	bool mtMode;

public:
	J2kEncoderAsync() {
		this->info = false;
		this->mtMode = false;
	};
	~J2kEncoderAsync(void) { };

	fastStatus_t Init(
		BaseOptions* baseOptions,
		MtResult* result,
		int threadId,
		void* specialParams
	)
	{
		this->options = *((J2kEncoderOptions*)baseOptions);
		info = baseOptions->Info;
		mtMode = result != nullptr;
		options.Info = false;

		CHECK_FAST(J2kEncoderBase::Init(options));

		CHECK_FAST(fastImportFromHostCreate(
			&hHostToDeviceAdapter,
			options.SurfaceFmt,

			options.MaxWidth,
			options.MaxHeight,

			&srcBuffer
		));

		CHECK_FAST(fastEncoderJ2kCreate(
			&hEncoder,
			&parameters,

			options.SurfaceFmt,
			options.MaxWidth,
			options.MaxHeight,
			options.BatchSize,

			srcBuffer
		));

		bool success = false;
		CHECK_FAST(fastEncoderJ2kIsInitialized(hEncoder, &success));
		if (!success) return FAST_INSUFFICIENT_DEVICE_MEMORY;

		size_t requestedMemSize = 0;
		size_t llComponentMemSize = 0;
		size_t componentMemSize = 0;
		CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hHostToDeviceAdapter, &componentMemSize));
		requestedMemSize += componentMemSize;
		CHECK_FAST(fastEncoderJ2kGetAllocatedGpuMemorySize(hEncoder, &llComponentMemSize));
		requestedMemSize += llComponentMemSize;

		const double gigabyte = 1024.0 * 1024.0 * 1024.0;
		if (mtMode && result != nullptr) {
			result->requestedMemSize = requestedMemSize / gigabyte;
		}
		else {
			printf("\nRequested GPU memory size: %.2lf GB\n", requestedMemSize / gigabyte);
		}
		return FAST_OK;
	}

	fastStatus_t Transform(
		Reader* imgs,
		BytestreamAsyncFileWriter<ManagedConstFastAllocator<1>>* j2ks,
		unsigned threadId,
		MtResult* result,
		volatile bool* terminate,
		void* specialParams
	)
	{
		double fullTime = 0.;

		const hostTimer_t encode_timer = hostTimerCreate();

		ManagedConstFastAllocator<1> alloc;
		const int imgSurfaceSize = options.MaxHeight * GetPitchFromSurface(surfaceFmt, options.MaxWidth) * sizeof(unsigned char);

		size_t totalFileSize = 0;
		size_t imageCount = 0;

		double processTimeAll = 0.0;
		double releaseTimeAll = 0.0;
		double allocTimeAll = 0.0;
		double writerTimeAll = 0.0;
		double readerTimeAll = 0.0;

		while (!(*terminate)) {
			hostTimerStart(encode_timer, info);
			auto imgBatch = imgs->ReadNextFileBatch(threadId);
			double getReaderTime = hostTimerEnd(encode_timer, info);

			if (imgBatch.IsEmpty())
				break;

			hostTimerStart(encode_timer, info);
			auto j2kBatch = j2ks->GetNextWriterBatch(threadId);
			double getWriterTime = hostTimerEnd(encode_timer, info);

			j2kBatch.SetFilltedItem(imgBatch.GetFilledItem());

			double processTime = 0.0;
			double releaseTime = 0.0;
			double allocTime = 0.0;
			for (int i = 0; i < imgBatch.GetFilledItem() && !(*terminate); i++) {
				hostTimerStart(encode_timer, info);
				auto img = imgBatch.At(i);
				auto j2k = j2kBatch.At(i);
				{
					j2k->inputFileName = img->inputFileName;
					j2k->outputFileName = img->outputFileName;
					j2k->data.reset(static_cast<unsigned char*>(alloc.allocate(imgSurfaceSize)));

					j2k->size = imgSurfaceSize;
					j2k->encoded = false;
					j2k->loadTimeMs = 0.;
				}
				allocTime += hostTimerEnd(encode_timer, info);

				hostTimerStart(encode_timer);
				{
					fastEncoderJ2kDynamicParameters_t dynamicParam = { 0 };
					{
						dynamicParam.targetStreamSize = 0;
						if (options.CompressionRatio > 1) {
							const size_t size = options.MaxWidth * GetNumberOfChannelsFromSurface(options.SurfaceFmt) * options.MaxHeight;
							dynamicParam.targetStreamSize = (long)floor((double)size / (double)options.CompressionRatio);
						}

						dynamicParam.quality = options.Quality;
						dynamicParam.writeHeader = !options.NoHeader;
					}

					CHECK_FAST(fastImportFromHostCopy(
						hHostToDeviceAdapter,

						img->data.get(),
						img->w,
						img->wPitch,
						img->h
					));

					CHECK_FAST(fastEncoderJ2kAddImageToBatch(
						hEncoder,
						&dynamicParam,

						img->w,
						img->h
					));
				}
				processTime += hostTimerEnd(encode_timer, info);

				imageCount++;
			}

			hostTimerStart(encode_timer);
			int unprocessedImagesCount = 0;
			fastEncoderJ2kUnprocessedImagesCount(hEncoder, &unprocessedImagesCount);
			if (unprocessedImagesCount > 0 && !(*terminate)) {
				fastEncoderJ2kReport_t report = { 0 };
				fastEncoderJ2kOutput_t output = { 0 };
				{
					auto j2k = j2kBatch.At(0);
					output.bufferSize = j2k->size;
					output.byteStream = j2k->data.get();

					CHECK_FAST(fastEncoderJ2kTransformBatch(hEncoder, &output, &report));
					j2k->size = output.streamSize;
				}


				int imagesLeft = unprocessedImagesCount - 1;
				for (int i = 1; i < unprocessedImagesCount && !(*terminate); i++) {
					auto j2k = j2kBatch.At(i);
					{
						output.bufferSize = j2k->size;
						output.byteStream = j2k->data.get();
					}
					CHECK_FAST(fastEncoderJ2kGetNextEncodedImage(hEncoder, &output, &report, &imagesLeft));
					j2k->size = output.streamSize;
				}
			}
			processTime += hostTimerEnd(encode_timer);

			hostTimerStart(encode_timer, info);
			for (int i = 0; i < imgBatch.GetFilledItem() && !(*terminate); i++) {
				auto img = imgBatch.At(i);
				img->ReleaseBuffer();
			}
			releaseTime += hostTimerEnd(encode_timer, info);

			processTimeAll += processTime;
			releaseTimeAll += releaseTime;
			allocTimeAll += allocTime;
			writerTimeAll += getWriterTime;
			readerTimeAll += getReaderTime;
		}

		j2ks->WriterFinished(threadId);

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

		hostTimerDestroy(encode_timer);

		return FAST_OK;
	}

	fastStatus_t Close()
	{
		CHECK_FAST(fastEncoderJ2kDestroy(hEncoder));
		CHECK_FAST(fastImportFromHostDestroy(hHostToDeviceAdapter));

		return FAST_OK;
	}
};

