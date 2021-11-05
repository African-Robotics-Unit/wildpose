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

#include <nvjpeg.h>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"

#include "Image.h"
#include "JpegEncoderSampleOptions.h"
#include "MultiThreadInfo.hpp"

#include "BatchedQueue.h"

#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"
#include "CollectionFastAllocator.hpp"

#include "AsyncFileReader.hpp"
#include "AsyncFileWriter.hpp"

#include "SurfaceTraits.hpp"

#include "string.h"

template <class Reader>
class NvjpegEncoderAsync {
private:
	nvjpegHandle_t nvjpeg_handle;

	nvjpegEncoderParams_t encode_params;
	nvjpegEncoderState_t encoder_state;

	bool info;

	JpegEncoderSampleOptions options;
	bool mtMode;

	nvjpegImage_t imgdesc;

public:
	NvjpegEncoderAsync() {
		this->info = false;
		this->mtMode = false;
	};
	~NvjpegEncoderAsync(void) { };

	fastStatus_t Init(
		BaseOptions* baseOptions,
		MtResult* result,
		int threadId,
		void* specialParams)
	{
		this->options = *((JpegEncoderSampleOptions*)baseOptions);
		info = baseOptions->Info;
		mtMode = result != nullptr;

		nvjpegStatus_t error = nvjpegCreateSimple(&nvjpeg_handle);
		if (error != NVJPEG_STATUS_SUCCESS) {
			fprintf(stderr, "nvjpegCreateSimple error code = %d\n", error);
			return FAST_INTERNAL_ERROR;
		}

		error = nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL);
		if (error != NVJPEG_STATUS_SUCCESS) {
			fprintf(stderr, "nvjpegEncoderStateCreate error code = %d\n", error);
			return FAST_INTERNAL_ERROR;
		}

		error = nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, NULL);
		if (error != NVJPEG_STATUS_SUCCESS) {
			fprintf(stderr, "nvjpegEncoderParamsCreate error code = %d\n", error);
			return FAST_INTERNAL_ERROR;
		}

		{
			memset(&imgdesc, 0, sizeof(nvjpegImage_t));
			const int imgSurfaceSize = options.MaxHeight * GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth) * sizeof(unsigned char);
			cudaMalloc((void**)&imgdesc.channel[0], imgSurfaceSize);
		}


		return FAST_OK;
	}

	fastStatus_t Transform(
		Reader* imgs,
		BytestreamAsyncFileWriter<ManagedConstFastAllocator<1>>* jfifs,
		unsigned threadId,
		MtResult* result,
		volatile bool* terminate,
		void* specialParams
	)
	{
		double fullTime = 0.;

		const hostTimer_t decode_timer = hostTimerCreate();

		ManagedConstFastAllocator<1> alloc;
		const int imgSurfaceSize = options.MaxHeight * GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth) * sizeof(unsigned char);

		size_t totalFileSize = 0;
		size_t imageCount = 0;

		double processTimeAll = 0.0;
		double releaseTimeAll = 0.0;
		double allocTimeAll = 0.0;
		double writerTimeAll = 0.0;
		double readerTimeAll = 0.0;

		fastGpuTimerHandle_t hostToDeviceTimer = NULL;
		if (options.BenchmarkInfo)
		{
			fastGpuTimerCreate(&hostToDeviceTimer);
		}

		while (!(*terminate)) {
			hostTimerStart(decode_timer, info);
			auto ppmBatch = imgs->ReadNextFileBatch(threadId);
			double getReaderTime = hostTimerEnd(decode_timer, info);

			if (ppmBatch.IsEmpty())
				break;

			hostTimerStart(decode_timer, info);
			auto jfifBatch = jfifs->GetNextWriterBatch(threadId);
			double getWriterTime = hostTimerEnd(decode_timer, info);

			jfifBatch.SetFilltedItem(ppmBatch.GetFilledItem());

			double processTime = 0.0;
			double releaseTime = 0.0;
			double allocTime = 0.0;
			for (int i = 0; i < ppmBatch.GetFilledItem() && !(*terminate); i++) {
				hostTimerStart(decode_timer, info);
				auto img = ppmBatch.At(i);
				auto jfif = jfifBatch.At(i);

				jfif->inputFileName = img->inputFileName;
				jfif->outputFileName = img->outputFileName;

				jfif->data.reset(static_cast<unsigned char*>(alloc.allocate(imgSurfaceSize)));
				jfif->size = imgSurfaceSize;

				{
					imgdesc.pitch[0] = img->wPitch;
				}

				allocTime += hostTimerEnd(decode_timer, info);

				if (img->w > options.MaxWidth ||
					img->h > options.MaxHeight) {
					fprintf(stderr, "Unsupported image size\n");
					continue;
				}

				hostTimerStart(decode_timer);
				{
					nvjpegStatus_t error = nvjpegEncoderParamsSetQuality(encode_params, options.JpegEncoder.Quality, NULL);
					if (error != NVJPEG_STATUS_SUCCESS) {
						fprintf(stderr, "nvjpegEncoderParamsSetQuality error code = %d\n", error);
						return FAST_INTERNAL_ERROR;
					}
					error = nvjpegEncoderParamsSetOptimizedHuffman(encode_params, 0 /*not-optimized version*/, NULL);
					if (error != NVJPEG_STATUS_SUCCESS) {
						fprintf(stderr, "nvjpegEncoderParamsSetOptimizedHuffman error code = %d\n", error);
						return FAST_INTERNAL_ERROR;
					}
					switch (options.JpegEncoder.SamplingFmt) {
					case FAST_JPEG_Y:
						error = nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_GRAY, NULL);
						break;
					case FAST_JPEG_444:
						error = nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_444, NULL);
						break;
					case FAST_JPEG_422:
						error = nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_422, NULL);
						break;
					case FAST_JPEG_420:
						error = nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_420, NULL);
						break;
					}
					if (options.SurfaceFmt == FAST_I8) {
						error = nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_GRAY, NULL);
					}
					if (error != NVJPEG_STATUS_SUCCESS) {
						fprintf(stderr, "nvjpegEncoderParamsSetSamplingFactors error code = %d\n", error);
						return FAST_INTERNAL_ERROR;
					}
					//nvjpegEncoderParamsSetEncoding(encode_params, nvjpegJpegEncoding_t etype, NULL);

					if (options.BenchmarkInfo)
					{
						fastGpuTimerStart(hostToDeviceTimer);
					}
					cudaError_t cudaError = cudaMemcpy2D(
						imgdesc.channel[0], imgdesc.pitch[0],
						img->data.get(), img->wPitch,
						img->w * GetNumberOfChannelsFromSurface(img->surfaceFmt), img->h,
						cudaMemcpyHostToDevice
					);
					if (cudaError != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy2D error code = %d\n", cudaError);
						return FAST_INTERNAL_ERROR;
					}

					if (options.BenchmarkInfo)
					{
						fastGpuTimerStop(hostToDeviceTimer);
					}

					if (options.SurfaceFmt == FAST_I8) {
						error = nvjpegEncodeYUV(
							nvjpeg_handle,
							encoder_state,
							encode_params,
							&imgdesc,
							NVJPEG_CSS_GRAY,
							img->w, img->h,
							NULL
						);
					}
					else {
						error = nvjpegEncodeImage(
							nvjpeg_handle,
							encoder_state,
							encode_params,
							&imgdesc,
							NVJPEG_INPUT_RGBI,
							img->w, img->h,
							NULL
						);
					}
					if (error != NVJPEG_STATUS_SUCCESS) {
						fprintf(stderr, "nvjpegEncodeImage error code = %d\n", error);
						return FAST_INTERNAL_ERROR;
					}

					error = nvjpegEncodeRetrieveBitstream(
						nvjpeg_handle,
						encoder_state,
						jfif->data.get(),
						&jfif->size,
						NULL
					);
					if (error != NVJPEG_STATUS_SUCCESS) {
						fprintf(stderr, "nvjpegEncodeRetrieveBitstream error code = %d\n", error);
						return FAST_INTERNAL_ERROR;
					}

					totalFileSize += img->w * img->h * GetNumberOfChannelsFromSurface(img->surfaceFmt);
				}
				processTime += hostTimerEnd(decode_timer);
				if (options.BenchmarkInfo)
				{
					float elapsedTime;
					fastGpuTimerGetTime(hostToDeviceTimer, &elapsedTime);
					processTime -= elapsedTime / 1000.0;
				}

				hostTimerStart(decode_timer, info);
				img->ReleaseBuffer();

				releaseTime += hostTimerEnd(decode_timer, info);
				imageCount++;
			}

			processTimeAll += processTime;
			releaseTimeAll += releaseTime;
			allocTimeAll += allocTime;
			writerTimeAll += getWriterTime;
			readerTimeAll += getReaderTime;
		}

		jfifs->WriterFinished(threadId);

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
			fastGpuTimerDestroy(hostToDeviceTimer);
		}


		return FAST_OK;
	}

	fastStatus_t Close()
	{
		nvjpegStatus_t error = nvjpegEncoderParamsDestroy(encode_params);
		if (error != NVJPEG_STATUS_SUCCESS) {
			fprintf(stderr, "nvjpegEncoderParamsDestroy error code = %d\n", error);
			return FAST_INTERNAL_ERROR;
		}

		error = nvjpegEncoderStateDestroy(encoder_state);
		if (error != NVJPEG_STATUS_SUCCESS) {
			fprintf(stderr, "nvjpegJpegStateDestroy error code = %d\n", error);
			return FAST_INTERNAL_ERROR;
		}
		error = nvjpegDestroy(nvjpeg_handle);
		if (error != NVJPEG_STATUS_SUCCESS) {
			fprintf(stderr, "nvjpegDestroy error code = %d\n", error);
			return FAST_INTERNAL_ERROR;
		}

		cudaFree(imgdesc.channel[0]);

		return FAST_OK;
	}
};
