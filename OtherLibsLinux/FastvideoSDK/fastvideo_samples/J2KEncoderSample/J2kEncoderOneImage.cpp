/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "J2kEncoderOneImage.h"

#include <ctime>
#include <cmath>
#include <cstdio>
#include <string.h>
#include <list>

#include <fastvideo_sdk.h>
#include <timing.hpp>
#include <supported_files.hpp>
#include <checks.h>
#include <helper_bytestream.hpp>

J2kEncoderOneImage::J2kEncoderOneImage(bool info) { this->info = info; };

J2kEncoderOneImage::~J2kEncoderOneImage(void) { };

fastStatus_t J2kEncoderOneImage::Init(J2kEncoderOptions &options) {
	CHECK_FAST(J2kEncoderBase::Init(options));
	if (batchSize != 1) {
		fprintf(stderr, "Incorrect max batch size\n");
		return FAST_INVALID_VALUE;
	}

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
	printf("Requested GPU memory size: %.2lf GB\n", requestedMemSize / gigabyte);
	
	return FAST_OK;
}

fastStatus_t J2kEncoderOneImage::Transform(std::list< Image<FastAllocator> > &images) {
	fastGpuTimerHandle_t hostToDeviceTimer = NULL;

	float elapsedTimeGpu = 0.;
	double totalTime = 0.;

	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
	}
	double totalFileSize = 0;
	
	const int MaxWidth = (options.MaxWidth == 0) ? (*images.begin()).w : options.MaxWidth;
	const int MaxHeight = (options.MaxHeight == 0) ? (*images.begin()).h : options.MaxHeight;

	unsigned maxOutputSize = static_cast<unsigned>(MaxWidth * GetNumberOfChannelsFromSurface(surfaceFmt) * MaxHeight * GetBytesPerChannelFromSurface(surfaceFmt) * MaximumSizeIncrease + 256);

	Bytestream<MallocAllocator> outputBytestream;
	{
		outputBytestream.size = maxOutputSize;
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

	for (auto i = images.begin(); i != images.end(); i++) {
		Image<FastAllocator> &img = *i;
		outputBytestream.outputFileName = img.outputFileName;

		double size = fileSize(img.inputFileName.c_str());
		totalFileSize += size * options.RepeatCount;

		fastEncoderJ2kDynamicParameters_t dynamicParam = { 0 };
		{
			dynamicParam.targetStreamSize = 0;
			if (options.CompressionRatio > 1)
				dynamicParam.targetStreamSize = (long)floor(size / options.CompressionRatio);

			dynamicParam.quality = options.Quality;
			dynamicParam.writeHeader = !options.NoHeader;
		}

		printf("Input image: %s (%dx%d pixels; %dx%d-bit channel(s)) - %.1f MB\n",
			img.inputFileName.c_str(), img.w, img.h, GetNumberOfChannelsFromSurface(img.surfaceFmt),
			img.bitsPerChannel, size / (1024.0f * 1024.0f));
		printf("Tile size: %dx%d\n", 
			(options.TileWidth == 0)?img.w: options.TileWidth, 
			(options.TileHeight == 0)?img.h: options.TileHeight);

		if (img.w > maxWidth || img.h > maxHeight) {
			fprintf(stderr, "ERROR: Image size exceeds the specified maximum size\n");
			continue;
		}
		
		for (int images_count = 0; images_count<options.RepeatCount; images_count++) {
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

			fastEncoderJ2kReport_t encodingReport = { 0 };
			{
				encodingReport.s8_write = 0;
			}

			CHECK_FAST(fastEncoderJ2kTransform(
				hEncoder,
				&dynamicParam,

				img.w,
				img.h,
				&output,
				&encodingReport
			));

			if (info) {
				double megabyte = 1024.0 * 1024.0;
				if (GetNumberOfChannelsFromSurface(img.surfaceFmt) == 3)
					printf("%7.2f ms 1) Preprocessing time (byte->int, DC-shift, MCT)\n", encodingReport.s1_preprocessing * 1000.0);
				else
					printf("%7.2f ms 1) Preprocessing time (byte->int, DC-shift)\n", encodingReport.s1_preprocessing * 1000.0);
				printf("%7.2f ms 2) DWT time\n", encodingReport.s2_dwt * 1000.0);
				printf("%7.2f ms 3) Tier-1 time (%d codeblocks)\n",
					encodingReport.s3_tier1 * 1000.0, encodingReport.codeblockCount);
				if (encodingReport.s4_pcrd >= 0)
					printf("%7.2f ms 4) PCRD time\n", encodingReport.s4_pcrd * 1000.0);
				else
					printf("           4) PCRD is disabled\n");
				printf("%7.2f ms 5) Buffers gathering time\n", encodingReport.s5_gathering * 1000.0);
				printf("%7.2f ms 6) GPU->CPU copy time (%.2lf MB, %.2lf MB/s)\n",
					encodingReport.s6_copy * 1000.0,
					encodingReport.copySize / megabyte,
					encodingReport.copySize / encodingReport.s6_copy / megabyte);
				printf("%7.2f ms 7) Tier-2 time\n", encodingReport.s7_tier2 * 1000.0);
				if (options.Discard)
					printf("(excluded) 8) Buffer write disabled; size = ");
				else
					printf("(excluded) 8) Buffer write time = %.2f ms; size = ", encodingReport.s8_write * 1000.0);
				if (encodingReport.outputSize < 1024)
					printf("%d bytes", encodingReport.outputSize);
				else
					printf("%.0f KB", encodingReport.outputSize / 1024.0f);

				int fsize = fileSize(img.inputFileName.c_str());
				printf(" (%.1f:1)\n", (float)fsize / encodingReport.outputSize);
				printf("%7.2f ms Total time\n", encodingReport.elapsedTime * 1000.0);

				totalTime += encodingReport.elapsedTime;
				printf("Encode time = %6.2f ms (%.0f MB/s)\n", encodingReport.elapsedTime * 1000.0, batchSize * fsize / (1024.0f * 1024.0f * encodingReport.elapsedTime));
			} else {
				totalTime += encodingReport.elapsedTime;
			}
			outputBytestream.size = output.streamSize;
			if (!options.Discard)
				CHECK_FAST(fvSaveBytestream(img.outputFileName, outputBytestream, false));
		}
	}

	const unsigned processedImages = static_cast<unsigned>(options.RepeatCount * images.size());
	double totalFileSizeMb = totalFileSize / (1024.0 * 1024.0);
	if (info) 
		printf("Total encode time for %d images = %.1f ms; %.0f MB/s; %.1f FPS;\n", processedImages,
			totalTime * 1000.0, totalFileSizeMb / totalTime, processedImages / totalTime);
	else
		printf("Total encode time excluding host-to-device transfer for %d images = %.1f ms; %.0f MB/s; %.1f FPS;\n", processedImages,
			totalTime * 1000.0, totalFileSizeMb / totalTime , processedImages / totalTime);
	

	if (info) {
		fastGpuTimerDestroy(hostToDeviceTimer);
	}

	return FAST_OK;
}

fastStatus_t J2kEncoderOneImage::Close(void) const {
	CHECK_FAST(fastEncoderJ2kDestroy(hEncoder));
	CHECK_FAST(fastImportFromHostDestroy(hHostToDeviceAdapter));

	return FAST_OK;
}
