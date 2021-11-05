/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "J2kEncoderBatch.h"

#include <ctime>
#include <cmath>
#include <cstdio>
#include <string.h>
#include <list>

#include "fastvideo_sdk.h"
#include "timing.hpp"
#include "supported_files.hpp"
#include "helper_bytestream.hpp"
#include "checks.h"

J2kEncoderBatch::J2kEncoderBatch(bool info, bool mtMode) {
	this->info = info;  this->mtMode = mtMode;
};
J2kEncoderBatch::~J2kEncoderBatch(void) { };

fastStatus_t J2kEncoderBatch::Init(J2kEncoderOptions &options, MtResult *result) {
	CHECK_FAST(J2kEncoderBase::Init(options));
	if (options.BatchSize == 1) {
		fprintf(stderr, "Unsupported batch size\n");
		return FAST_INVALID_VALUE;
	}
	batchSize = options.BatchSize;
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
	} else {
		printf("Requested GPU memory size: %.2lf GB\n", requestedMemSize / gigabyte);
	}

	return FAST_OK;
}

fastStatus_t J2kEncoderBatch::Transform(std::list< Image<FastAllocator> > &images, MtResult *result) {
	fastGpuTimerHandle_t hostToDeviceTimer = NULL;
	
	float elapsedTimeGpu = 0.;
	double totalTime = 0.;

	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
	}

	unsigned maxOutputSize = static_cast<unsigned>(options.MaxWidth * GetNumberOfChannelsFromSurface(surfaceFmt) * options.MaxHeight * GetBytesPerChannelFromSurface(surfaceFmt) * MaximumSizeIncrease + 256);
	Bytestream<MallocAllocator> outputBytestream;
	{
		outputBytestream.size = maxOutputSize;
		outputBytestream.inputFileName = "";
		outputBytestream.encoded = false;

		MallocAllocator alloc;
		outputBytestream.data.reset((unsigned char*)alloc.allocate(outputBytestream.size));
		outputBytestream.loadTimeMs = 0.;
	}

	fastEncoderJ2kOutput_t output = { 0 };
	{
		output.bufferSize = maxOutputSize;
		output.byteStream = outputBytestream.data.get();
	}

	long long totalFileSize = 0;
	for (auto i = images.begin(); i != images.end(); ++i) {
		Image<FastAllocator> &img = *i;
		long long size = fileSize(img.inputFileName.c_str());
		totalFileSize += size * options.RepeatCount;

		fastEncoderJ2kDynamicParameters_t dynamicParam = { 0 };
		{
			dynamicParam.targetStreamSize = 0;
			if (options.CompressionRatio > 1)
				dynamicParam.targetStreamSize = (long)floor(size / options.CompressionRatio);

			dynamicParam.quality = options.Quality;
			dynamicParam.writeHeader = !options.NoHeader;
		}
		if (!mtMode && info)
			printf("Input image: %s (%dx%d pixels; %dx%d-bit channel(s)) - %.1f MB\n",
				img.inputFileName.c_str(), img.w, img.h, GetNumberOfChannelsFromSurface(img.surfaceFmt),
				img.bitsPerChannel, size / (1024.0f * 1024.0f));

		if (img.w > maxWidth || img.h > maxHeight) {
			fprintf(stderr, "ERROR: Image size exceeds the specified maximum size\n");
			continue;
		}

		int freeSlots = 0;
		for (int images_count = 0; images_count < options.RepeatCount; images_count++) {
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

				totalTime += elapsedTimeGpu/1000.0;
			}

			fastEncoderJ2kFreeSlotsInBatch(hEncoder, &freeSlots);
			if (freeSlots <= 0) {
				fastEncoderJ2kReport_t report = { 0 };
				CHECK_FAST(fastEncoderJ2kTransformBatch(hEncoder, &output, &report));

				auto fileName = batchFileName.begin();
				outputBytestream.size = output.streamSize;
				if (!options.Discard)
					CHECK_FAST(fvSaveBytestream(*fileName++, outputBytestream, false));

				int imagesLeft = 0;
				do {
					CHECK_FAST(fastEncoderJ2kGetNextEncodedImage(hEncoder, &output, &report, &imagesLeft));

					outputBytestream.size = output.streamSize;
					if (!options.Discard)
						CHECK_FAST(fvSaveBytestream(*fileName++, outputBytestream, false));
				} while (imagesLeft != 0);

				batchFileName.clear();
				totalTime += report.elapsedTime;
			}
			CHECK_FAST(fastEncoderJ2kAddImageToBatch(
				hEncoder,
				&dynamicParam,

				img.w,
				img.h
			));
			batchFileName.push_back(img.outputFileName);
		}
	}

	{
		int unprocessedImagesCount = 0;
		fastEncoderJ2kUnprocessedImagesCount(hEncoder, &unprocessedImagesCount);
		if (unprocessedImagesCount > 0) // Process the last non-complete batch
		{
			fastEncoderJ2kReport_t report = { 0 };
			CHECK_FAST(fastEncoderJ2kTransformBatch(hEncoder, &output, &report));

			auto fileName = batchFileName.begin();
			outputBytestream.size = output.streamSize;
			if (!options.Discard)
				CHECK_FAST(fvSaveBytestream(*fileName++, outputBytestream, false));

			int imagesLeft = unprocessedImagesCount - 1;
			while (imagesLeft != 0) {
				CHECK_FAST(fastEncoderJ2kGetNextEncodedImage(hEncoder, &output, &report, &imagesLeft));
				outputBytestream.size = output.streamSize;
				if (!options.Discard)
					CHECK_FAST(fvSaveBytestream(*fileName++, outputBytestream, false));
			}

			totalTime += report.elapsedTime;
			batchFileName.clear();
		}
	}

	double totalFileSizeMb = totalFileSize / (1024.0 * 1024.0);
	const int processedImages = options.RepeatCount * static_cast<int>(images.size());
	if (mtMode) {
		result->totalTime = totalTime;
		result->totalFileSize = static_cast<double>(totalFileSize);
	} else
		if (info)
			printf("Total encode time for %d images = %.1f ms; %.0f MB/s; %.1f FPS;\n", processedImages,
				totalTime * 1000.0, totalFileSizeMb / totalTime , processedImages / totalTime);
		else
			printf("Total encode time excluding host-to-device transfer for %d images = %.1f ms; %.0f MB/s; %.1f FPS;\n", processedImages,
				totalTime * 1000.0, totalFileSizeMb / totalTime, processedImages / totalTime);

	if (info) {
		fastGpuTimerDestroy(hostToDeviceTimer);
	}

	return FAST_OK;
}

fastStatus_t J2kEncoderBatch::Close(void) {
	fastStatus_t ret = FAST_OK;

	CHECK_FAST(fastEncoderJ2kDestroy(hEncoder));
	CHECK_FAST(fastImportFromHostDestroy(hHostToDeviceAdapter));

	return ret;
}
