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
#include "fastvideo_decoder_j2k.h"

#include "Image.h"
#include "J2kDecoderOptions.h"
#include "J2kDecoderBase.h"
#include "MultiThreadInfo.hpp"

#include "BatchedQueue.h"

#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"
#include "AsyncFileReader.hpp"
#include "AsyncFileWriter.hpp"

#include "PreloadJ2k.hpp"

template <class Reader>
class J2kDecoderAsync : public J2kDecoderBase {
private:
	fastDecoderJ2kHandle_t hDecoder;
	fastExportToHostHandle_t hDeviceToHostAdapter;

	fastDeviceSurfaceBufferHandle_t dstBuffer;
	fastDeviceSurfaceBufferHandle_t stubBuffer;

	fastJ2kImageInfo_t baseImageInfo;
	bool info;
	bool mtMode;

public:
	J2kDecoderAsync() {
		this->info = false;
		this->mtMode = false;
	};
	~J2kDecoderAsync(void) { };

	fastStatus_t Init(
		BaseOptions* baseOptions,
		MtResult* result,
		int threadId,
		void* specialParams	)
	{
		this->options = *((J2kDecoderOptions*)baseOptions);
		info = baseOptions->Info;
		mtMode = result != nullptr;

		baseImageInfo = {};
		PreloadJ2k(baseOptions->InputPath, baseOptions->IsFolder, &baseImageInfo);
		CHECK_FAST(J2kDecoderBase::Init(options, &baseImageInfo));

		parameters.enableMemoryReallocation = options.IsFolder;
		if (options.IsFolder)
			parameters.maxStreamSize = GetBufferSizeFromSurface(options.SurfaceFmt, options.MaxWidth, options.MaxHeight);

		unsigned outputWidth = options.MaxWidth, outputHeight = options.MaxHeight;
		if (parameters.windowWidth)
			outputWidth = std::min(outputWidth, (unsigned)parameters.windowWidth);
		if (parameters.windowHeight)
			outputHeight = std::min(outputHeight, (unsigned)parameters.windowHeight);

		CHECK_FAST(fastDeviceSurfaceBufferStubCreate(options.SurfaceFmt, outputWidth, outputHeight, &stubBuffer));

		size_t requestedMemSize = 0;
		CHECK_FAST(fastExportToHostExclusiveCreate(&hDeviceToHostAdapter, &surfaceFmt, stubBuffer));
		{
			size_t componentMemSize = 0;
			CHECK_FAST(fastExportToHostGetAllocatedGpuMemorySize(hDeviceToHostAdapter, &componentMemSize));
			requestedMemSize += componentMemSize;
		}

		CHECK_FAST(fastDecoderJ2kCreate(&hDecoder, &parameters, options.SurfaceFmt, options.MaxWidth, options.MaxHeight, options.BatchSize, &dstBuffer));
		{
			size_t llComponentMemSize = 0;
			CHECK_FAST(fastDecoderJ2kGetAllocatedGpuMemorySize(hDecoder, &llComponentMemSize));
			requestedMemSize += llComponentMemSize;
		}

		CHECK_FAST(fastExportToHostChangeSrcBuffer(hDeviceToHostAdapter, dstBuffer));

		CHECK_FAST(fastDeviceSurfaceBufferStubDestroy(&stubBuffer));

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
		Reader* j2ks,
		PortableAsyncFileWriter<ManagedConstFastAllocator<0>>* imgs,
		unsigned threadId,
		MtResult* result,
		volatile bool* terminate,
		void* specialParams
	)
	{
		double fullTime = 0.;
		const hostTimer_t decode_timer = hostTimerCreate();

		ManagedConstFastAllocator<0> alloc;
		const int imgSurfaceSize = options.MaxHeight * GetPitchFromSurface(surfaceFmt, options.MaxWidth) * sizeof(unsigned char);

		size_t totalFileSize = 0;
		size_t imageCount = 0;

		double processTimeAll = 0.0;
		double releaseTimeAll = 0.0;
		double allocTimeAll = 0.0;
		double writerTimeAll = 0.0;
		double readerTimeAll = 0.0;

		while (!(*terminate)) {
			hostTimerStart(decode_timer, info);
			auto jfifBatch = j2ks->ReadNextFileBatch(threadId);
			double getReaderTime = hostTimerEnd(decode_timer, info);

			if (jfifBatch.IsEmpty())
				break;

			hostTimerStart(decode_timer, info);
			auto imgBatch = imgs->GetNextWriterBatch(threadId);
			double getWriterTime = hostTimerEnd(decode_timer, info);

			imgBatch.SetFilltedItem(jfifBatch.GetFilledItem());

			double processTime = 0.0;
			double releaseTime = 0.0;
			double allocTime = 0.0;
			for (int i = 0; i < jfifBatch.GetFilledItem() && !(*terminate); i++) {
				hostTimerStart(decode_timer, info);
				auto jfif = jfifBatch.At(i);
				auto img = imgBatch.At(i);
				{
					img->inputFileName = jfif->inputFileName;
					img->outputFileName = jfif->outputFileName;
					img->data.reset(static_cast<unsigned char*>(alloc.allocate(imgSurfaceSize)));
				}
				allocTime += hostTimerEnd(decode_timer, info);

				hostTimerStart(decode_timer);
				CHECK_FAST(fastDecoderJ2kAddImageToBatch(hDecoder, jfif->data.get(), jfif->size));
				processTime += hostTimerEnd(decode_timer, info);

				imageCount++;
			}

			hostTimerStart(decode_timer);
			{
				int unprocessedImagesCount = 0;
				CHECK_FAST(fastDecoderJ2kUnprocessedImagesCount(hDecoder, &unprocessedImagesCount));
				if (unprocessedImagesCount > imgBatch.GetFilledItem()) {
					fprintf(stderr, "Image batch less than unprocessed image count\n");
					continue;
				}

				fastDecoderJ2kReport_t report = { 0 };
				CHECK_FAST(fastDecoderJ2kTransformBatch(hDecoder, &report));

				int imagesLeft = unprocessedImagesCount;
				for (int i = 0; i < unprocessedImagesCount && !(*terminate); i++) {
					auto img = imgBatch.At(i);
					img->surfaceFmt = surfaceFmt;
					img->bitsPerChannel = report.bitsPerChannel;

					img->w = report.width;
					img->wPitch = (unsigned)GetPitchFromSurface(surfaceFmt, img->w);
					img->h = report.height;

					fastExportParameters_t exportParameters = { };
					exportParameters.convert = options.ConvertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
					CHECK_FAST(fastExportToHostCopy(
						hDeviceToHostAdapter,

						img->data.get(),
						img->w,
						img->wPitch,
						img->h,

						&exportParameters
					));

					CHECK_FAST(fastDecoderJ2kGetNextDecodedImage(hDecoder, &report, &imagesLeft));

					totalFileSize += img->w * img->h * GetNumberOfChannelsFromSurface(img->surfaceFmt);;
				}
			}
			processTime += hostTimerEnd(decode_timer);

			hostTimerStart(decode_timer, info);
			for (int i = 0; i < jfifBatch.GetFilledItem() && !(*terminate); i++) {
				auto jfif = jfifBatch.At(i);
				jfif->ReleaseBuffer();
			}
			releaseTime += hostTimerEnd(decode_timer, info);

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
		CHECK_FAST(fastDecoderJ2kDestroy(hDecoder));
		CHECK_FAST(fastExportToHostDestroy(hDeviceToHostAdapter));
		return FAST_OK;
	}
};

