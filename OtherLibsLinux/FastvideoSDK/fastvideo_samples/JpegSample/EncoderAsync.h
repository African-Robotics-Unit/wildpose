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
#include "JpegEncoderSampleOptions.h"
#include "MultiThreadInfo.hpp"

#include "BatchedQueue.h"

#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"
#include "AsyncFileReader.hpp"
#include "AsyncFileWriter.hpp"

#include <libexif/exif-content.h>
#include "EnumToStringSdk.h"
#include "helper_jpeg.hpp"
#include "helper_exif.hpp"

#include "SurfaceTraits.hpp"

template <class Reader>
class EncoderAsync {
private:
	fastImportFromHostHandle_t hImportFromHost;
	fastJpegEncoderHandle_t hEncoder;

	fastDeviceSurfaceBufferHandle_t srcBuffer;

	fastJpegExifSection_t *exifSections;
	unsigned exifSectionsCount;

	bool mtMode;

	bool info;
	bool benchmarkInfo;
	JpegEncoderSampleOptions options;

private:
	fastStatus_t InitExif()
	{
		unsigned char* exifBytestream;
		unsigned int   exifBytestreamLen;

		ExifData* exifData = exif_data_new();

		exif_data_unset_option(exifData, EXIF_DATA_OPTION_FOLLOW_SPECIFICATION);
		exif_data_set_option(exifData, EXIF_DATA_OPTION_DONT_CHANGE_MAKER_NOTE);
		exif_data_set_byte_order(exifData, EXIF_BYTE_ORDER_MOTOROLA);

		const char* make = "fastvideo encoder";
		ExifEntry* entry = fastExifCreateAsciiTag(exifData, EXIF_IFD_0, EXIF_TAG_MAKE, static_cast<unsigned>(strlen(make) + 1));
		strcpy((char*)entry->data, make);

		time_t timer;
		time(&timer);
		struct tm* time = localtime(&timer);
		char dataStr[20];
		sprintf(static_cast<char*>(dataStr), "%4d:%02d:%02d %02d:%02d:%02d", time->tm_year + 1900, time->tm_mon + 1, time->tm_mday, time->tm_hour, time->tm_min, time->tm_sec); /*YYYY:MM:DD HH:MM:SS*/
		entry = fastExifCreateAsciiTag(exifData, EXIF_IFD_0, EXIF_TAG_DATE_TIME, static_cast<unsigned>(strlen(dataStr) + 1));
		strcpy((char*)entry->data, dataStr);

		char buffer[30];
		strcpy(static_cast<char*>(buffer), "ASCII");
		buffer[7] = 0;
		strcpy(static_cast<char*>(buffer) + 8, make);

		entry = fastExifCreateAsciiTag(exifData, EXIF_IFD_EXIF, EXIF_TAG_USER_COMMENT, static_cast<unsigned>(strlen(make) + 1 + 8));
		memcpy((char*)entry->data, buffer, strlen(make) + 1 + 8);

		exif_data_save_data(exifData, &exifBytestream, &exifBytestreamLen);

		exifSections = new fastJpegExifSection_t[1];
		exifSections[0].exifCode = EXIF_SECTION_CODE;
		exifSections[0].exifLength = exifBytestreamLen;
		exifSections[0].exifData = new char[exifBytestreamLen];
		memcpy(exifSections[0].exifData, exifBytestream, exifBytestreamLen);

		exifSectionsCount = 1;

		if (exifBytestream)
			free(exifBytestream);

		exif_data_free(exifData);

		return FAST_OK;
	}

public:
	EncoderAsync() {
		this->info = false;
		this->benchmarkInfo = false;
		this->mtMode = false;
		hEncoder = NULL; hImportFromHost = NULL;
		exifSections = nullptr;
		exifSectionsCount = 0;
	};
	~EncoderAsync(void) { };

	fastStatus_t Init(
		BaseOptions* baseOptions,
		MtResult *result,
		int threadId,
		void* specialParams	)
	{
		this->options = *((JpegEncoderSampleOptions*)baseOptions);
		info = baseOptions->Info;
		benchmarkInfo = baseOptions->BenchmarkInfo;

		mtMode = result != nullptr;

		if (!options.JpegEncoder.noExif) {
			InitExif();
		}

		CHECK_FAST(fastImportFromHostCreate(
			&hImportFromHost,

			options.SurfaceFmt,
			options.MaxWidth,
			options.MaxHeight,

			&srcBuffer
		));
		if (options.SurfaceFmt == FAST_BGR8) {
			options.SurfaceFmt = FAST_RGB8;
		}

		CHECK_FAST(fastJpegEncoderCreate(
			&hEncoder,

			options.MaxWidth,
			options.MaxHeight,

			srcBuffer
		));

		size_t requestedMemSpace = 0;
		size_t tmp = 0;

		CHECK_FAST(fastJpegEncoderGetAllocatedGpuMemorySize(hEncoder, &tmp));
		requestedMemSpace += tmp;

		CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
		requestedMemSpace += tmp;

		const double megabyte = 1024.0 * 1024.0;
		if (mtMode && result != nullptr) {
			result->requestedMemSize = requestedMemSpace / megabyte;
		}
		else
			printf("Requested GPU memory space: %.2f MB\n\n", requestedMemSpace / megabyte);

		return FAST_OK;
	}

	fastStatus_t Transform(
		Reader* imgs,
		JpegAsyncFileWriter<ManagedConstFastAllocator<1>>* jfifs,
		unsigned threadId,
		MtResult *result,
		volatile bool* terminate,
		void* specialParams
	)

	{
		double fullTime = 0.;

		const hostTimer_t hostTimer = hostTimerCreate();

		ManagedConstFastAllocator<1> alloc;
		const int imgSurfaceSize = options.MaxHeight * GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth) * sizeof(unsigned char);

		size_t totalFileSize = 0;
		size_t imageCount = 0;

		fastGpuTimerHandle_t hostToDeviceTimer = NULL;
		fastGpuTimerHandle_t jpegEncoderTimer = NULL;

		if (benchmarkInfo)
		{
			fastGpuTimerCreate(&hostToDeviceTimer);
			fastGpuTimerCreate(&jpegEncoderTimer);
		}

		double processTimeAll = 0.0;
		double releaseTimeAll = 0.0;
		double allocTimeAll = 0.0;
		double writerTimeAll = 0.0;
		double readerTimeAll = 0.0;
		double componentTimeAll = 0.0;

		fastJpegQuantState_t* quantState = (fastJpegQuantState_t*)specialParams;

		while (!(*terminate)) {
			hostTimerStart(hostTimer, info);
			auto ppmBatch = imgs->ReadNextFileBatch(threadId);
			double getReaderTime = hostTimerEnd(hostTimer, info);

			if (ppmBatch.IsEmpty())
				break;

			hostTimerStart(hostTimer, info);
			auto jfifBatch = jfifs->GetNextWriterBatch(threadId);
			double getWriterTime = hostTimerEnd(hostTimer, info);

			jfifBatch.SetFilltedItem(ppmBatch.GetFilledItem());

			double processTime = 0.0;
			double releaseTime = 0.0;
			double allocTime = 0.0;
			double componentTime = 0.0;
			for (int i = 0; i < ppmBatch.GetFilledItem() && !(*terminate); i++) {
				hostTimerStart(hostTimer, info);
				auto img = ppmBatch.At(i);
				auto jfif = jfifBatch.At(i);

				jfif->inputFileName = img->inputFileName;
				jfif->outputFileName = img->outputFileName;

				jfif->bytestream.reset(static_cast<unsigned char*>(alloc.allocate(imgSurfaceSize)));

				memset(&jfif->info, 0, sizeof(fastJfifInfo_t));
				jfif->info.restartInterval = options.JpegEncoder.RestartInterval;
				jfif->info.jpegFmt = options.JpegEncoder.SamplingFmt;
				jfif->info.bitsPerChannel = GetBitsPerChannelFromSurface(options.SurfaceFmt);

				jfif->info.exifSections = exifSections;
				jfif->info.exifSectionsCount = exifSectionsCount;

				jfif->info.width = img->w;
				jfif->info.height = img->h;
				allocTime += hostTimerEnd(hostTimer, info);

				if (img->w > options.MaxWidth ||
					img->h > options.MaxHeight) {
					fprintf(stderr, "Unsupported image size\n");
					continue;
				}

				hostTimerStart(hostTimer);
				if (benchmarkInfo)
					fastGpuTimerStart(hostToDeviceTimer);

				CHECK_FAST(fastImportFromHostCopy(
					hImportFromHost,

					img->data.get(),
					img->w,
					img->wPitch,
					img->h
				));

				if (benchmarkInfo) {
					fastGpuTimerStop(hostToDeviceTimer);
					fastGpuTimerStart(jpegEncoderTimer);
				}

				if (quantState != NULL) {
					CHECK_FAST(fastJpegEncodeWithQuantTable(
						hEncoder,
						quantState,
						&jfif->info
					));
				}
				else {
					CHECK_FAST(fastJpegEncode(
						hEncoder,

						options.JpegEncoder.Quality,
						jfif->GetFastInfo()
					));
				}
				if (benchmarkInfo)
					fastGpuTimerStop(jpegEncoderTimer);

				totalFileSize += img->w * img->h * GetNumberOfChannelsFromSurface(img->surfaceFmt);
				double processTimeImage = hostTimerEnd(hostTimer);
				processTime += processTimeImage;

				hostTimerStart(hostTimer, info);
				img->ReleaseBuffer();
				releaseTime += hostTimerEnd(hostTimer, info);
				imageCount++;

				if (benchmarkInfo) {
					float elapsedEncodeGpu = 0.0, elapsedHostToDevice = 0.0;
					fastGpuTimerGetTime(jpegEncoderTimer, &elapsedEncodeGpu);
					fastGpuTimerGetTime(hostToDeviceTimer, &elapsedHostToDevice);

					double elapsedTotalEncodeTime = processTimeImage * 1000.0 - (elapsedEncodeGpu + elapsedHostToDevice);
					componentTime = ((double)elapsedEncodeGpu + ((elapsedTotalEncodeTime > 0.0) ? elapsedTotalEncodeTime : 0.0)) / 1000.0;
				}
			}

			processTimeAll += processTime;
			releaseTimeAll += releaseTime;
			allocTimeAll += allocTime;
			writerTimeAll += getWriterTime;
			readerTimeAll += getReaderTime;
			componentTimeAll += componentTime;
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
			result->componentTime = componentTimeAll;
		}

		hostTimerDestroy(hostTimer);

		if (benchmarkInfo)
		{
			fastGpuTimerDestroy(hostToDeviceTimer);
			fastGpuTimerDestroy(jpegEncoderTimer);
		}

		return FAST_OK;
	}

	fastStatus_t Close()
	{
		CHECK_FAST(fastJpegEncoderDestroy(hEncoder));
		CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));

		for (unsigned i = 0; i < exifSectionsCount; i++) {
			free(exifSections[i].exifData);
		}

		if (exifSections != NULL) {
			free(exifSections);
		}

		return FAST_OK;
	}
};
