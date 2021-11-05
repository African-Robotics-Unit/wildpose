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

#include <nvjpeg.h>

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

template <class Reader>
class NvjpegDecoderBatchedAsync {
private:
	nvjpegJpegState_t nvjpeg_state;
	nvjpegHandle_t nvjpeg_handle;

	fastSurfaceFormat_t surfaceFmt;

	bool info;

	JpegDecoderSampleOptions options;
	bool mtMode;

	nvjpegImage_t* gpuOutImage;

public:
	NvjpegDecoderBatchedAsync() {
		this->info = false;
		this->mtMode = false;
	};
	~NvjpegDecoderBatchedAsync(void) { };

	fastStatus_t Init(
		BaseOptions* baseOptions,
		MtResult* result,
		int threadId,
		void* specialParams
	)
	{
		this->options = *((JpegDecoderSampleOptions*)baseOptions);
		info = baseOptions->Info;
		surfaceFmt = baseOptions->SurfaceFmt;
		mtMode = result != nullptr;

		//nvjpegStatus_t error = nvjpegCreateSimple(&nvjpeg_handle);

		//2k_wild.1280.420.1.jpg

		nvjpegStatus_t error = nvjpegCreateEx(
			NVJPEG_BACKEND_GPU_HYBRID,
			NULL,
			NULL,
			0,
			&nvjpeg_handle);

		if (error != NVJPEG_STATUS_SUCCESS) {
			fprintf(stderr, "nvjpegCreateSimple error code = %d\n", error);
			return FAST_INTERNAL_ERROR;
		}

		error = nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);
		if (error != NVJPEG_STATUS_SUCCESS) {
			fprintf(stderr, "nvjpegJpegStateCreate error code = %d\n", error);
			return FAST_INTERNAL_ERROR;
		}
		error = nvjpegDecodeBatchedInitialize(
			nvjpeg_handle, nvjpeg_state,
			options.BatchSize, /*options.NumberOfThreads*/ 4,
			options.SurfaceFmt == FAST_I8 ? NVJPEG_OUTPUT_Y : NVJPEG_OUTPUT_RGBI
		);
		if (error != NVJPEG_STATUS_SUCCESS) {
			fprintf(stderr, "nvjpegDecodeBatchedInitialize error code = %d\n", error);
			return FAST_INTERNAL_ERROR;
		}

		gpuOutImage = new nvjpegImage_t[options.BatchSize];
		for (int i = 0; i < options.BatchSize; i++)
		{
			for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
				gpuOutImage[i].channel[c] = NULL;
				gpuOutImage[i].pitch[c] = 0;
			}

			unsigned pitch = GetPitchFromSurface(surfaceFmt, baseOptions->MaxWidth);
			cudaMalloc((void**)&gpuOutImage[i].channel[0], pitch * baseOptions->MaxHeight);
		}

		return FAST_OK;
	}

	fastStatus_t Transform(
		Reader* bytestreams,
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

		unsigned cnt = 0;

		fastGpuTimerHandle_t deviceToHostTimer = NULL;

		if (options.BenchmarkInfo)
		{
			fastGpuTimerCreate(&deviceToHostTimer);
		}

		while (!(*terminate)) {
			hostTimerStart(decode_timer, info);
			auto bytestreamBatch = bytestreams->ReadNextFileBatch(threadId);
			double getReaderTime = hostTimerEnd(decode_timer, info);

			if (bytestreamBatch.IsEmpty())
				break;

			hostTimerStart(decode_timer, info);
			auto ppmBatch = imgs->GetNextWriterBatch(threadId);
			double getWriterTime = hostTimerEnd(decode_timer, info);

			ppmBatch.SetFilltedItem(bytestreamBatch.GetFilledItem());

			double processTime = 0.0;
			double releaseTime = 0.0;
			double allocTime = 0.0;

			unsigned char** src = new unsigned char* [bytestreamBatch.GetFilledItem()];
			size_t* srcLength = new size_t[bytestreamBatch.GetFilledItem()];

			hostTimerStart(decode_timer, info);

			if (bytestreamBatch.GetFilledItem() != options.BatchSize)
			{
				nvjpegStatus_t error = nvjpegDecodeBatchedInitialize(
					nvjpeg_handle, nvjpeg_state,
					bytestreamBatch.GetFilledItem(), /*options.NumberOfThreads*/ 4,
					options.SurfaceFmt == FAST_I8 ? NVJPEG_OUTPUT_Y : NVJPEG_OUTPUT_RGBI
				);
				if (error != NVJPEG_STATUS_SUCCESS) {
					fprintf(stderr, "nvjpegDecodeBatchedInitialize error code = %d\n", error);
					return FAST_INTERNAL_ERROR;
				}
			}

			for (int i = 0; i < bytestreamBatch.GetFilledItem() && !(*terminate); i++) {
				auto bytestream = bytestreamBatch.At(i);
				auto img = ppmBatch.At(i);

				img->inputFileName = bytestream->inputFileName;
				img->outputFileName = bytestream->outputFileName;
				img->data.reset(static_cast<unsigned char*>(alloc.allocate(imgSurfaceSize)));

				int numberOfChannels = 0;
				nvjpegChromaSubsampling_t subsampling;
				int w[NVJPEG_MAX_COMPONENT] = { 0 }, h[NVJPEG_MAX_COMPONENT] = { 0 };
				nvjpegStatus_t error = nvjpegGetImageInfo(
					nvjpeg_handle,
					bytestream->data.get(),
					bytestream->size,
					&numberOfChannels,
					&subsampling,
					w, h
				);
				if (error != NVJPEG_STATUS_SUCCESS) {
					fprintf(stderr, "nvjpegGetImageInfo error code = %d\n", error);
					return FAST_INTERNAL_ERROR;
				}

				img->w = w[0]; img->h = h[0];
				img->surfaceFmt = surfaceFmt = IdentifySurface(8, numberOfChannels);
				img->wPitch = GetPitchFromSurface(surfaceFmt, img->w);

				if (img->w > options.MaxWidth || img->h > options.MaxHeight) {
					fprintf(stderr, "Unsupported image size\n");
					continue;
				}

				src[i] = bytestream->data.get();
				srcLength[i] = bytestream->size;

				gpuOutImage[i].pitch[0] = GetPitchFromSurface(surfaceFmt, img->w);
			}
			allocTime += hostTimerEnd(decode_timer, info);

			hostTimerStart(decode_timer);

			nvjpegStatus_t nvjpegError = nvjpegDecodeBatched(
				nvjpeg_handle,
				nvjpeg_state,
				src, srcLength,
				gpuOutImage,
				cudaStreamPerThread
			);
			if (nvjpegError != NVJPEG_STATUS_SUCCESS) {
				fprintf(stderr, "nvjpegDecodeBatched error code = %d\n", nvjpegError);
				return FAST_INTERNAL_ERROR;
			}

			if (options.BenchmarkInfo)
			{
				fastGpuTimerStart(deviceToHostTimer);
			}

			for (int i = 0; i < bytestreamBatch.GetFilledItem() && !(*terminate); i++) {
				auto bytestream = bytestreamBatch.At(i);
				auto img = ppmBatch.At(i);

				cudaError_t cudaError = cudaMemcpy2D(
					img->data.get(), img->wPitch,
					gpuOutImage[i].channel[0], gpuOutImage[i].pitch[0],
					img->w * GetNumberOfChannelsFromSurface(surfaceFmt), img->h,
					cudaMemcpyDeviceToHost
				);
				if (cudaError != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy2D error code = %d\n", cudaError);
					return FAST_INTERNAL_ERROR;
				}

				const unsigned channelCount = options.SurfaceFmt == FAST_I8 ? 1 : 3;
				totalFileSize += img->w * img->h * channelCount;
			}
			if (options.BenchmarkInfo)
			{
				fastGpuTimerStop(deviceToHostTimer);
			}

			double pt = hostTimerEnd(decode_timer);

			if (options.BenchmarkInfo)
			{
				float elapsedTime;
				fastGpuTimerGetTime(deviceToHostTimer, &elapsedTime);
				pt -= elapsedTime / 1000.0;
			}
			if (cnt > 1)
				processTime += pt;
			else if (cnt == 1)
				processTime += 2 * pt;


			for (int i = 0; i < bytestreamBatch.GetFilledItem() && !(*terminate); i++) {
				auto bytestream = bytestreamBatch.At(i);
				hostTimerStart(decode_timer, info);
				bytestream->ReleaseBuffer();
				releaseTime += hostTimerEnd(decode_timer, info);
				imageCount++;
			}

			delete[] src;
			delete[] srcLength;

			processTimeAll += processTime;
			releaseTimeAll += releaseTime;
			allocTimeAll += allocTime;
			writerTimeAll += getWriterTime;
			readerTimeAll += getReaderTime;
			cnt++;
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
			result->componentTime = processTimeAll;
		}

		hostTimerDestroy(decode_timer);
		if (options.BenchmarkInfo)
		{
			fastGpuTimerDestroy(deviceToHostTimer);
		}

		return FAST_OK;
	}

	fastStatus_t Close()
	{
		nvjpegStatus_t error = nvjpegJpegStateDestroy(nvjpeg_state);
		if (error != NVJPEG_STATUS_SUCCESS) {
			fprintf(stderr, "nvjpegJpegStateDestroy error code = %d\n", error);
			return FAST_INTERNAL_ERROR;
		}
		error = nvjpegDestroy(nvjpeg_handle);
		if (error != NVJPEG_STATUS_SUCCESS) {
			fprintf(stderr, "nvjpegDestroy error code = %d\n", error);
			return FAST_INTERNAL_ERROR;
		}

		delete[] gpuOutImage;
		for (int i = 0; i < options.BatchSize; i++)
		{
			cudaFree(gpuOutImage[i].channel[0]);
		}
		return FAST_OK;
	}
};

