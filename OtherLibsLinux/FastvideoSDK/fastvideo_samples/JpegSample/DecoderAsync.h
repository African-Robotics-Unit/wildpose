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
#include <vector_types.h>
#include "FastAllocator.h"

#include "fastvideo_sdk.h"

#include "Image.h"
#include "JpegDecoderSampleOptions.h"
#include "MultiThreadInfo.hpp"

#include "BatchedQueue.h"

#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"
#include "AsyncFileReader.hpp"
#include "AsyncFileWriter.hpp"

#include "EnumToStringSdk.h"
#include "helper_jpeg.hpp"
#include "SurfaceTraits.hpp"

template <class Reader>
class DecoderAsync {
private:
	fastJpegDecoderHandle_t hDecoder;
	fastExportToHostHandle_t hDeviceToHostAdapter;

	fastDeviceSurfaceBufferHandle_t dstBuffer;

	fastSurfaceFormat_t surfaceFmt;

	bool benchmarkInfo;
	bool info;

	JpegDecoderSampleOptions options;
	bool mtMode;

public:
	DecoderAsync() {
		this->info = false;
		this->benchmarkInfo = false;
		this->mtMode = false;
	};
	~DecoderAsync(void) { };

	fastStatus_t Init(
		BaseOptions *baseOptions,
		MtResult *result,
		int threadId,
		void* specialParams	)
	{
		this->options = *((JpegDecoderSampleOptions*)baseOptions);
		info = baseOptions->Info;
		benchmarkInfo = baseOptions->BenchmarkInfo;
		mtMode = result != nullptr;

		CHECK_FAST(fastJpegDecoderCreate(
			&hDecoder,

			options.SurfaceFmt == FAST_BGR8 ? FAST_RGB8 : options.SurfaceFmt,
			options.MaxWidth,
			options.MaxHeight,
			true,

			&dstBuffer
		));

		CHECK_FAST(fastExportToHostCreate(
			&hDeviceToHostAdapter,

			&surfaceFmt,

			dstBuffer
		));

		size_t requestedMemSpace = 0;
		CHECK_FAST(fastJpegDecoderGetAllocatedGpuMemorySize(hDecoder, &requestedMemSpace));
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
		PortableAsyncFileWriter<ManagedConstFastAllocator<0>>* imgs,
		unsigned threadId, 
		MtResult *result,
		volatile bool* terminate,
		void* specialParams
	)
	{
		double fullTime = 0.;

		const hostTimer_t decode_timer = hostTimerCreate();

		fastGpuTimerHandle_t deviceToHostTimer = NULL;
		fastGpuTimerHandle_t jpegDecoderTimer = NULL;

		if (benchmarkInfo)
		{
			fastGpuTimerCreate(&deviceToHostTimer);
			fastGpuTimerCreate(&jpegDecoderTimer);
		}

		ManagedConstFastAllocator<0> alloc;
		const int imgSurfaceSize = options.MaxHeight * GetPitchFromSurface(surfaceFmt, options.MaxWidth) * sizeof(unsigned char);

		size_t totalFileSize = 0;
		size_t imageCount = 0;

		double processTimeAll = 0.0;
		double componentTimeAll = 0.0;
		double releaseTimeAll = 0.0;
		double allocTimeAll = 0.0;
		double writerTimeAll = 0.0;
		double readerTimeAll = 0.0;

		while (!(*terminate)) {
			hostTimerStart(decode_timer, info);
			auto jfifBatch = jfifs->ReadNextFileBatch(threadId);
			double getReaderTime = hostTimerEnd(decode_timer, info);

			if (jfifBatch.IsEmpty())
				break;

			hostTimerStart(decode_timer, info);
			auto ppmBatch = imgs->GetNextWriterBatch(threadId);
			double getWriterTime = hostTimerEnd(decode_timer, info);

			ppmBatch.SetFilltedItem(jfifBatch.GetFilledItem());

			double processTime = 0.0;
			double processTimeImage = 0.0;
			double componentTime = 0.0;
			double releaseTime = 0.0;
			double allocTime = 0.0;
			for (int i = 0; i < jfifBatch.GetFilledItem() && !(*terminate); i++) {
				hostTimerStart(decode_timer, info);
				auto jfif = jfifBatch.At(i);
				auto img = ppmBatch.At(i);

				img->inputFileName = jfif->inputFileName;
				img->outputFileName = jfif->outputFileName;
				img->data.reset(static_cast<unsigned char*>(alloc.allocate(imgSurfaceSize)));

				allocTime += hostTimerEnd(decode_timer, info);

				hostTimerStart(decode_timer);
				if (jfif->info.width > options.MaxWidth ||
					jfif->info.height > options.MaxHeight) {
					fprintf(stderr, "Unsupported image size\n");
					continue;
				}

				img->w = jfif->info.width;
				img->h = jfif->info.height;
				img->surfaceFmt = surfaceFmt;
				img->wPitch = GetPitchFromSurface(surfaceFmt, img->w);

				if (benchmarkInfo)
					fastGpuTimerStart(jpegDecoderTimer);

				CHECK_FAST(fastJpegDecode(
					hDecoder,

					jfif->GetFastInfo()
				));

				if (benchmarkInfo) {
					fastGpuTimerStop(jpegDecoderTimer);
					fastGpuTimerStart(deviceToHostTimer);
				}

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

				if (benchmarkInfo)
					fastGpuTimerStop(deviceToHostTimer);

				const unsigned channelCount = (jfif->info.jpegFmt == FAST_JPEG_Y) ? 1 : 3;
				totalFileSize += jfif->info.width * jfif->info.height * channelCount;

				processTimeImage = hostTimerEnd(decode_timer);
				processTime += processTimeImage;

				hostTimerStart(decode_timer, info);
				jfif->ReleaseBuffer();
				releaseTime += hostTimerEnd(decode_timer, info);
				imageCount++;

				if (benchmarkInfo)
				{
					float elapsedDecodeGpu = 0.0, elapsedDeviceToHost = 0.0;
					fastGpuTimerGetTime(jpegDecoderTimer, &elapsedDecodeGpu);
					fastGpuTimerGetTime(deviceToHostTimer, &elapsedDeviceToHost);

					double elapsedTotalDecodeTime = processTimeImage * 1000.0 - (elapsedDecodeGpu + elapsedDeviceToHost);
					componentTime = ((double)elapsedDecodeGpu + ((elapsedTotalDecodeTime > 0.0) ? elapsedTotalDecodeTime : 0.0)) / 1000.0;
				}
			}

			processTimeAll += processTime;
			releaseTimeAll += releaseTime;
			allocTimeAll += allocTime;
			writerTimeAll += getWriterTime;
			readerTimeAll += getReaderTime;
			componentTimeAll += componentTime;
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
			result->componentTime = componentTimeAll;
		}

		hostTimerDestroy(decode_timer);
		if (benchmarkInfo)
		{
			fastGpuTimerDestroy(deviceToHostTimer);
			fastGpuTimerDestroy(jpegDecoderTimer);
		}

		return FAST_OK;
	}

	fastStatus_t Close()
	{
		CHECK_FAST(fastJpegDecoderDestroy(hDecoder));
		CHECK_FAST(fastExportToHostDestroy(hDeviceToHostAdapter));

		return FAST_OK;
	}
};

