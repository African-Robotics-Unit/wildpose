/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <cstdio>

#include "DecoderMerger.h"
#include "checks.h"
#include "supported_files.hpp"
#include "BayerExifInfo.hpp"
#include "helper_jpeg.hpp"

fastStatus_t DecoderMerger::Init(BaseOptions &options, unsigned maxRestoredWidth, unsigned maxRestoredHeight) {
	if (options.SurfaceFmt != FAST_I8 && options.SurfaceFmt != FAST_I12 && options.SurfaceFmt != FAST_I16) {
		fprintf(stderr, "Unsupported source format\n");
		return FAST_INVALID_FORMAT;
	}

	this->options = options;
	if (options.SurfaceFmt == FAST_I8) {
		CHECK_FAST(fastJpegDecoderCreate(
			&hDecoder,

			options.SurfaceFmt,
			options.MaxWidth,
			options.MaxHeight,
			true,
			&decoderBuffer
		));
	} else {
		CHECK_FAST(fastJpegCpuDecoderCreate(
			&hDecoderCpu,

			options.SurfaceFmt,
			options.MaxWidth,
			options.MaxHeight,

			&decoderBuffer
		));
	}

	CHECK_FAST(fastBayerMergerCreate(
		&hBayerMerger,

		maxRestoredWidth,
		maxRestoredHeight,

		decoderBuffer,
		&bayerMergerBuffer
	));

	CHECK_FAST(fastExportToHostCreate(
		&hExportToHost,

		&surfaceFmt,

		bayerMergerBuffer
	));
	options.SurfaceFmt = surfaceFmt;

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(h_Result.reset((unsigned char *)alloc.allocate(GetPitchFromSurface(surfaceFmt, maxRestoredWidth) * maxRestoredHeight)));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;

	if (hDecoder != NULL) {
		CHECK_FAST(fastJpegDecoderGetAllocatedGpuMemorySize(hDecoder, &tmp));
		requestedMemSpace += tmp;
	}

	if (hDecoderCpu != NULL) {
		CHECK_FAST(fastJpegCpuGetAllocatedGpuMemorySize(hDecoderCpu, &tmp));
		requestedMemSpace += tmp;
	}

	CHECK_FAST(fastBayerMergerGetAllocatedGpuMemorySize(hBayerMerger, &tmp));
	requestedMemSpace += tmp;

	printf("Requested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t DecoderMerger::Close() const {
	if (hDecoder != NULL) {
		CHECK_FAST(fastJpegDecoderDestroy(hDecoder));
	}
	if (hDecoderCpu != NULL) {
		CHECK_FAST(fastJpegCpuDecoderDestroy(hDecoderCpu));
	}
	CHECK_FAST(fastBayerMergerDestroy(hBayerMerger));
	CHECK_FAST(fastExportToHostDestroy(hExportToHost));

	return FAST_OK;
}

fastStatus_t DecoderMerger::Transform(std::list< Bytestream<FastAllocator> > &inputImages) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	unsigned restoredWidth, restoredHeight;

	fastGpuTimerHandle_t decoderTimer = NULL;
	fastGpuTimerHandle_t bayerMergerTimer = NULL;
	fastGpuTimerHandle_t exportToHostTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&decoderTimer);
		fastGpuTimerCreate(&bayerMergerTimer);
		fastGpuTimerCreate(&exportToHostTimer);
	}

	for (auto i = inputImages.begin(); i != inputImages.end(); i++) {
		jfifInfo.bytestreamSize = (*i).size;
		CHECK_FAST(fastMalloc((void **)&jfifInfo.h_Bytestream, jfifInfo.bytestreamSize));

		CHECK_FAST(fastJfifLoadFromMemory(
			(*i).data.get(),
			(*i).size,

			&jfifInfo
		));

		printf("Input image: %s\nInput image size: %dx%d pixels, %d bits\n\n",
			(*i).inputFileName.c_str(), jfifInfo.width, jfifInfo.height, jfifInfo.bitsPerChannel
		);
		if (options.MaxWidth < jfifInfo.width ||
			options.MaxHeight < jfifInfo.height) {
			fprintf(stderr, "No decoder initialized with these parameters\n");
			continue;
		}

		bool bayerExifExist = false;
		if (jfifInfo.exifSections != NULL) {
			for (unsigned j = 0; j < jfifInfo.exifSectionsCount; j++) {
				fastBayerPattern_t bayerFormat;
				if (ParseSplitterExif(&jfifInfo.exifSections[j], bayerFormat, restoredWidth, restoredHeight) == FAST_OK) {
					bayerExifExist = true;
				}

				free(jfifInfo.exifSections[j].exifData);
			}

			free(jfifInfo.exifSections);
		}

		if (!bayerExifExist) {
			fprintf(stderr, "Incorrect JPEG (%s): debayer parameters in EXIF sections was not found\n", (*i).inputFileName.c_str());
			continue;
		}

		if (info) {
			fastGpuTimerStart(decoderTimer);
		}

		if (hDecoder != NULL) {
			CHECK_FAST(fastJpegDecode(
				hDecoder,

				&jfifInfo
			));
		} else {
			CHECK_FAST(fastJpegCpuDecode(
				hDecoderCpu,

				(*i).data.get(),
				(*i).size,

				&jfifInfo
			));
		}

		if (info) {
			fastGpuTimerStop(decoderTimer);
			fastGpuTimerGetTime(decoderTimer, &elapsedTimeGpu);
			printf("Decode time = %.2f ms\n", elapsedTimeGpu);

			fullTime += elapsedTimeGpu;

			fastGpuTimerStart(bayerMergerTimer);
		}

		CHECK_FAST(fastBayerMergerMerge(
			hBayerMerger,

			restoredWidth,
			restoredHeight
		));

		if (info) {
			fastGpuTimerStop(bayerMergerTimer);
			fastGpuTimerGetTime(bayerMergerTimer, &elapsedTimeGpu);
			printf("Merge time = %.2f ms\n", elapsedTimeGpu);

			fullTime += elapsedTimeGpu;

			fastGpuTimerStart(exportToHostTimer);
		}

		fastExportParameters_t exportParameters = { };
		exportParameters.convert = options.ConvertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
		CHECK_FAST(fastExportToHostCopy(
			hExportToHost,

			h_Result.get(),
			restoredWidth,
			GetPitchFromSurface(surfaceFmt, restoredWidth),
			restoredHeight,

			&exportParameters
		));

		if (info) {
			fastGpuTimerStop(exportToHostTimer);
			fastGpuTimerGetTime(exportToHostTimer, &elapsedTimeGpu);

			fullTime += elapsedTimeGpu;
			printf("Device-to-host transfer = %.2f ms\n\n", elapsedTimeGpu);
		}

		CHECK_FAST(fvSaveImageToFile(
			(char *)(*i).outputFileName.c_str(),
			h_Result,
			surfaceFmt,
			restoredHeight,
			restoredWidth,
			GetPitchFromSurface(surfaceFmt, restoredWidth),
			false
		));

		printf("Output image: %s\nOutput image size: %dx%d pixels\n\n", (*i).outputFileName.c_str(), restoredWidth, restoredHeight);
	}

	if (info) {
		printf("Total for all images = %.2f ms\n", fullTime);

		fastGpuTimerDestroy(decoderTimer);
		fastGpuTimerDestroy(bayerMergerTimer);
		fastGpuTimerDestroy(exportToHostTimer);
	}

	return FAST_OK;
}
