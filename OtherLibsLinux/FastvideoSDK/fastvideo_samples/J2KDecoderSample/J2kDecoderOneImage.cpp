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

#include "J2kDecoderOneImage.h"
#include "J2kDecoderOptions.h"
#include "EnumToStringJ2kDecoder.h"
#include <algorithm>

J2kDecoderOneImage::J2kDecoderOneImage() {
    decoder = nullptr;
}

fastStatus_t J2kDecoderOneImage::Init(J2kDecoderOptions &options, fastJ2kImageInfo_t *sampleImage) {
	CHECK_FAST(J2kDecoderBase::Init(options, sampleImage));

	unsigned outputWidth = options.MaxWidth, outputHeight = options.MaxHeight;
	if (parameters.windowWidth)
		outputWidth = std::min(outputWidth, (unsigned) parameters.windowWidth);
	if (parameters.windowHeight)
		outputHeight = std::min(outputHeight, (unsigned) parameters.windowHeight);

	CHECK_FAST(fastDeviceSurfaceBufferStubCreate(options.SurfaceFmt, outputWidth, outputHeight, &stub));

	size_t requestedMemSize = 0;

	CHECK_FAST(fastExportToHostExclusiveCreate(&adapter, &surfaceFmt, stub));
	{
		size_t componentMemSize = 0;
		CHECK_FAST(fastExportToHostGetAllocatedGpuMemorySize(adapter, &componentMemSize));
		requestedMemSize += componentMemSize;
	}

	CHECK_FAST(fastDecoderJ2kCreate(
		&decoder,
		&parameters,
		options.SurfaceFmt, options.MaxWidth, options.MaxHeight,
		options.BatchSize,
		&buffer
	));
	{
		size_t llComponentMemSize = 0;
		CHECK_FAST(fastDecoderJ2kGetAllocatedGpuMemorySize(decoder, &llComponentMemSize));
		requestedMemSize += llComponentMemSize;
	}
	
	CHECK_FAST(fastExportToHostChangeSrcBuffer(adapter, buffer));

	CHECK_FAST(fastDeviceSurfaceBufferStubDestroy(&stub));
	
	const double gigabyte = 1024.0 * 1024.0 * 1024.0;
	printf("\nRequested GPU memory size: %.2lf GB\n",
		requestedMemSize / gigabyte
	);

	return FAST_OK;
}

fastStatus_t J2kDecoderOneImage::Transform(std::list< Bytestream< FastAllocator > > &inputImgs, std::list< Image<FastAllocator > > &outputImgs) const {
	double totalTime = 0.;
	const unsigned processedImages =  static_cast<unsigned>(inputImgs.size() * options.RepeatCount);
    fastGpuTimerHandle_t deviceToHostTimer = nullptr;

	if (options.Info) {
		fastGpuTimerCreate(&deviceToHostTimer);
	}

	fastDecoderJ2kReport_t report = { 0 };

	fastExportParameters_t exportParameters = { };
	exportParameters.convert = options.ConvertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;

	FastAllocator alloc;
	Image<FastAllocator> img;
	img.data.reset(static_cast<unsigned char *>(alloc.allocate(options.MaxHeight * GetPitchFromSurface(surfaceFmt, options.MaxWidth) * sizeof(unsigned char))));

	for (auto ii = inputImgs.begin(); ii != inputImgs.end(); ++ii) {
		img.inputFileName = (*ii).inputFileName;
		img.outputFileName = (*ii).outputFileName;

		for (int i = 0; i < options.RepeatCount; i++) {
			CHECK_FAST(fastDecoderJ2kTransform(decoder, (*ii).data.get(), (*ii).size, &report));

			img.w = report.width;
            img.wPitch = (unsigned)GetPitchFromSurface(surfaceFmt, img.w);
			img.h = report.height;
			img.surfaceFmt = surfaceFmt;
			img.bitsPerChannel = report.bitsPerChannel;

			totalTime += report.elapsedTime;

			if (!options.Discard) {
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

					totalTime += elapsedTimeGpu / 1000.0;
				}

                try
                {
				outputImgs.push_back(img);
			}
                catch (std::bad_alloc&)
                {
                    return FAST_INSUFFICIENT_HOST_MEMORY;
                }
            }

			if (options.Info && report.codeblockCount > 0) {
				printf("Input image : %s (%dx%d pixels; %d %d-bit channel(s))\n", (*ii).inputFileName.c_str(), img.w, img.h, report.channels, report.bitsPerChannel);
				printf("Tile count: %d (%dx%d)\n", report.tileCount, report.tilesX, report.tilesY);
			
				if (options.IsEnabledWindow) {
					printf("Window mode enabled: start position - (%d, %d), window size = %dx%d pixels\n",
						options.WindowLeftTopCoordsX, options.WindowLeftTopCoordsY,
						options.WindowWidth, options.WindowHeight
					);
				} else {
					printf("Window mode disabled: decode full image\n");
				}

				printf("Progression order: %s\n", EnumToString(report.progressionType));
				printf("Resolution levels: %d\n", report.resolutionLevels);
				printf("Codeblock size: %dx%d\n", report.cbX, report.cbY);
				printf("Wavelet filter: %s\n", EnumToString(report.dwtType));
				printf("MCT type: %s\n", EnumToString(report.mctType));

                if (options.Tier2Threads > 1 && report.tileCount > 0) printf("WARNING: The measured time of each stage can be far from the real time due to multithreading mode.\n");

				const double megabyte = 1024.0 * 1024.0;
                printf("%7.2lf ms 1) Tier-2 time (%d codeblocks)\n", report.s1_tier2 * 1000.0, report.codeblockCount);
                printf("%7.2lf ms 2) CPU->GPU copy time (%.2lf MB, %.2lf MB/s)\n",
					report.s2_copy * 1000.0,
					report.copyToGpu_size / megabyte,
					report.copyToGpu_size == 0 ? 0 : (report.copyToGpu_size / report.s2_copy / megabyte));
                printf("%7.2lf ms 3) Tier-1 time\n", report.s3_tier1 * 1000.0);
                printf("%7.2lf ms 4) DWT time\n", report.s6_dwt * 1000.0);
                printf("%7.2lf ms 5) Postprocessing time (MCT, DC-shift)\n", report.s7_postprocessing * 1000.0);
				printf("              Output size = ");
				if (report.outStreamSize < 1024)
                    printf("%lld bytes", report.outStreamSize);
				else
                    printf("%.0lf KB", report.outStreamSize / 1024.0);
                printf(" (%.1lf:1)\n", report.outStreamSize == 0 ? 0 : (double)report.outStreamSize / report.inStreamSize);
			}
		}
	}

	if (options.Info && !options.Discard)
		printf("Total decode time for %d images = %.1f ms; %.1f FPS;\n", processedImages,
			totalTime * 1000.0, processedImages / totalTime);
	else
		printf("Total decode time excluding device-to-host transfer for %d images = %.1f ms; %.1f FPS;\n", processedImages,
			totalTime * 1000.0, processedImages / totalTime);

	if (options.Info) {
		fastGpuTimerDestroy(deviceToHostTimer);
	}

	return FAST_OK;
}

fastStatus_t J2kDecoderOneImage::Close() const {
	CHECK_FAST(fastDecoderJ2kDestroy(decoder));
	CHECK_FAST(fastExportToHostDestroy(adapter));
	return FAST_OK;
}
