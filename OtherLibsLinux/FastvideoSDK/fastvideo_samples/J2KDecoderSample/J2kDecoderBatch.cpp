/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "checks.h"
#include "SurfaceTraits.hpp"

#include "J2kDecoderBatch.h"
#include "J2kDecoderOptions.h"

fastStatus_t J2kDecoderBatch::TransformAndExtractBatch(
	Image<FastAllocator>& img,
	std::list<Image<FastAllocator> > *outputImgs
) {
	fastDecoderJ2kReport_t report = { 0 };
	CHECK_FAST(fastDecoderJ2kTransformBatch(decoder, &report));

	// Extract all decoded images
	int imagesLeft = 1; /* at least one image in batch*/
	do {
		if (!options.Discard) {
			fastExportParameters_t exportParameters = { };
			exportParameters.convert = options.ConvertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;

			img.surfaceFmt = surfaceFmt;
			img.bitsPerChannel = report.bitsPerChannel;

			img.w = report.width;
			img.wPitch = (unsigned)GetPitchFromSurface(surfaceFmt, img.w);
			img.h = report.height;

			if (options.Info) {
				fastGpuTimerStart(deviceToHostTimer);
			}

			CHECK_FAST(fastExportToHostCopy(
				adapter,

				img.data.get(),
				img.w,
				img.wPitch,
				img.h,

				&exportParameters
			));

			if (options.Info) {
				float elapsedTimeGpu = 0.;
				fastGpuTimerStop(deviceToHostTimer);
				fastGpuTimerGetTime(deviceToHostTimer, &elapsedTimeGpu);

				totalInternalTime += elapsedTimeGpu / 1000.0;
			}
			try {
				outputImgs->push_back(img);
			} catch (std::bad_alloc&) {
				return FAST_INSUFFICIENT_HOST_MEMORY;
			}
		}
		if (imagesLeft == 0)
			break;
		CHECK_FAST(fastDecoderJ2kGetNextDecodedImage(decoder, &report, &imagesLeft));
	} while (report.inStreamSize != 0);

	totalInternalTime += report.elapsedTime;
	return FAST_OK;
}

J2kDecoderBatch::J2kDecoderBatch(bool mtMode) {
    decoder = nullptr;
	this->mtMode = mtMode;
    deviceToHostTimer = nullptr;
}

fastStatus_t J2kDecoderBatch::Init(J2kDecoderOptions &options, fastJ2kImageInfo_t *sampleImage, MtResult *result) {
	if (options.Info) {
		fastGpuTimerCreate(&deviceToHostTimer);
	}

	CHECK_FAST(J2kDecoderBase::Init(options, sampleImage));

	unsigned outputWidth = options.MaxWidth, outputHeight = options.MaxHeight;
	if (parameters.windowWidth)
		outputWidth = std::min(outputWidth, (unsigned)parameters.windowWidth);
	if (parameters.windowHeight)
		outputHeight = std::min(outputHeight, (unsigned)parameters.windowHeight);

	CHECK_FAST(fastDeviceSurfaceBufferStubCreate(options.SurfaceFmt, outputWidth, outputHeight, &stub));

	size_t requestedMemSize = 0;

	CHECK_FAST(fastExportToHostExclusiveCreate(&adapter, &surfaceFmt, stub));
	{
		size_t componentMemSize = 0;
		CHECK_FAST(fastExportToHostGetAllocatedGpuMemorySize(adapter, &componentMemSize));
		requestedMemSize += componentMemSize;
	}

	CHECK_FAST(fastDecoderJ2kCreate(&decoder, &parameters, options.SurfaceFmt, options.MaxWidth, options.MaxHeight, options.BatchSize, &buffer));
	{
		size_t llComponentMemSize = 0;
		CHECK_FAST(fastDecoderJ2kGetAllocatedGpuMemorySize(decoder, &llComponentMemSize));
		requestedMemSize += llComponentMemSize;
	}
		
	CHECK_FAST(fastExportToHostChangeSrcBuffer(adapter, buffer));

	CHECK_FAST(fastDeviceSurfaceBufferStubDestroy(&stub));

	const double gigabyte = 1024.0 * 1024.0 * 1024.0;
	if (mtMode && result != nullptr) {
		result->requestedMemSize = requestedMemSize / gigabyte;
	} else {
		printf("\nRequested GPU memory size: %.2lf GB\n", requestedMemSize / gigabyte);
	}

	return FAST_OK;
}

fastStatus_t J2kDecoderBatch::Transform(std::list< Bytestream< FastAllocator > > &inputImgs, std::list< Image<FastAllocator > > &outputImgs, MtResult *result) {
	FastAllocator alloc;
	Image<FastAllocator> img;
	img.data.reset(static_cast<unsigned char *>(alloc.allocate(options.MaxHeight * GetPitchFromSurface(surfaceFmt, options.MaxWidth) * sizeof(unsigned char))));

	totalInternalTime = 0.;
	const unsigned processedImages = static_cast<unsigned>(inputImgs.size() * options.RepeatCount);

	for (auto ii = inputImgs.begin(); ii != inputImgs.end(); ++ii) {
		img.inputFileName = (*ii).inputFileName;
		img.outputFileName = (*ii).outputFileName;

		int freeSlots = 0;
		for (int i = 0; i < options.RepeatCount; i++) {
			CHECK_FAST(fastDecoderJ2kAddImageToBatch(decoder, (*ii).data.get(), (*ii).size));
			CHECK_FAST(fastDecoderJ2kFreeSlotsInBatch(decoder, &freeSlots));
			if (freeSlots == 0) {
				CHECK_FAST(TransformAndExtractBatch(img, &outputImgs));
			}
		}
	}

	int unprocessedImagesCount = 0;
	CHECK_FAST(fastDecoderJ2kUnprocessedImagesCount(decoder, &unprocessedImagesCount));
	if (unprocessedImagesCount > 0) { // Process the last non-complete batch
		CHECK_FAST(TransformAndExtractBatch(img, &outputImgs));
	}

	if (mtMode) {
		result->totalTime = totalInternalTime;
	} else {
		if (options.Info && !options.Discard)
			printf("Total decode time for %d images = %.1f ms; %.1f FPS;\n", processedImages,
				totalInternalTime * 1000.0, processedImages / totalInternalTime);
		else
			printf("Total decode time excluding device-to-host transfer for %d images = %.1f ms; %.1f FPS;\n", processedImages,
				totalInternalTime * 1000.0, processedImages / totalInternalTime);
	}

	return FAST_OK;
}

fastStatus_t J2kDecoderBatch::Close() const {
	CHECK_FAST(fastDecoderJ2kDestroy(decoder));
	CHECK_FAST(fastExportToHostDestroy(adapter));
	if (options.Info) {
		fastGpuTimerDestroy(deviceToHostTimer);
	}
	return FAST_OK;
}
