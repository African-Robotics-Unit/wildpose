/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "stdio.h"

#include "DecoderCpu.h"
#include "timing.hpp"
#include "checks.h"
#include "supported_files.hpp"

inline unsigned DecoderCpu::uDivUp(unsigned a, unsigned b) {
	return (a / b) + (a % b != 0);
}

fastStatus_t DecoderCpu::Init(JpegDecoderSampleOptions &options) {
	this->options = options;

	if (options.SurfaceFmt != FAST_I12 && options.SurfaceFmt != FAST_RGB12) {
		return FAST_UNSUPPORTED_SURFACE;
	}

	fastSdkParametersHandle_t sdkParameters;
	CHECK_FAST(fastGetSdkParametersHandle(&sdkParameters));
	CHECK_FAST(fastJpegCpuDecoderLibraryInit(sdkParameters));

	CHECK_FAST(fastJpegCpuDecoderCreate(
		&hDecoder,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&dstBuffer
	));

	CHECK_FAST(fastExportToHostCreate(
		&hDeviceToHostAdapter,

		&surfaceFmt,

		dstBuffer
	));

	size_t requestedMemSpace = 0;
	CHECK_FAST(fastJpegCpuGetAllocatedGpuMemorySize(hDecoder, &requestedMemSpace));
	printf("Requested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t DecoderCpu::Close() const {
	CHECK_FAST(fastJpegCpuDecoderDestroy(hDecoder));
	CHECK_FAST(fastExportToHostDestroy(hDeviceToHostAdapter));

	return FAST_OK;
}

fastStatus_t DecoderCpu::Decode(std::list<Bytestream<FastAllocator> > &inputImgs, std::list< Image<FastAllocator> > &outputImgs) {
	hostTimer_t host_timer = NULL;
	hostTimer_t decode_timer = NULL;
	fastGpuTimerHandle_t deviceToHostTimer = NULL;
	fastGpuTimerHandle_t jpegDecoderTimer = NULL;

	float fullTime = 0.;

	if (info) {
		host_timer = hostTimerCreate();
		decode_timer = hostTimerCreate();
		CHECK_FAST(fastGpuTimerCreate(&deviceToHostTimer));
		fastGpuTimerCreate(&jpegDecoderTimer);
	}

	for (auto i = inputImgs.begin(); i != inputImgs.end(); ++i) {
		Image<FastAllocator> img;
		img.inputFileName = (*i).inputFileName;
		img.outputFileName = (*i).outputFileName;

		if (info) {
			hostTimerStart(decode_timer);
		}

		CHECK_FAST(fastJpegCpuDecode(
			hDecoder,

			(*i).data.get(),
			(*i).size,

			&jfifInfo
		));

		if (info) {
			fastGpuTimerStop(jpegDecoderTimer);
			CHECK_FAST(fastGpuTimerStart(deviceToHostTimer));
		}

		img.w = jfifInfo.width;
		img.h = jfifInfo.height;
		img.wPitch = GetPitchFromSurface(surfaceFmt, img.w);
		img.surfaceFmt = surfaceFmt;
		img.bitsPerChannel = 0;
		FastAllocator alloc;
		CHECK_FAST_ALLOCATION(img.data.reset((unsigned char *)alloc.allocate(img.h * img.wPitch)));

		fastExportParameters_t exportParameters = { };
		exportParameters.convert = options.ConvertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
		CHECK_FAST(fastExportToHostCopy(
			hDeviceToHostAdapter,

			img.data.get(),
			img.w,
			img.wPitch,
			img.h,

			&exportParameters
		));

		if (info) {
			float elapsedDecodeGpu = 0.;
			float elapsedDeviceToHost = 0.;

			CHECK_FAST(fastGpuTimerStop(deviceToHostTimer));
			float elapsedTotalDecodeTime = (float)hostTimerEnd(decode_timer)*1000.0f;
			fastGpuTimerGetTime(jpegDecoderTimer, &elapsedDecodeGpu);
			CHECK_FAST(fastGpuTimerGetTime(deviceToHostTimer, &elapsedDeviceToHost));

			elapsedTotalDecodeTime = elapsedTotalDecodeTime - elapsedDecodeGpu - elapsedDeviceToHost;
			const float elapsedDecode = elapsedDecodeGpu + ((elapsedTotalDecodeTime > 0.0f) ? elapsedTotalDecodeTime : 0.0f);

			printf("Decode time (includes host-to-device transfer) = %.2f ms\n", elapsedDecode);
			printf("Device-To-Host transfer = %.2f ms\n\n", elapsedDeviceToHost);

			fullTime += elapsedDecode;
			fullTime += elapsedDeviceToHost;
		}

		printf("Output image: %s\n\n", img.outputFileName.c_str());
		outputImgs.push_back(img);

		for (unsigned j = 0; j < jfifInfo.exifSectionsCount; j++) {
			free(jfifInfo.exifSections[j].exifData);
			jfifInfo.exifSections[j].exifData = NULL;
		}

		if (jfifInfo.exifSections != NULL) {
			free(jfifInfo.exifSections);
			jfifInfo.exifSections = NULL;
			jfifInfo.exifSectionsCount = 0;
		}
	}

	if (info) {
		printf("Total time for all images = %.2f ms (without HDD I/O)\n", fullTime);
		hostTimerDestroy(host_timer);
		CHECK_FAST(fastGpuTimerDestroy(deviceToHostTimer));
		fastGpuTimerDestroy(jpegDecoderTimer);
	}

	return FAST_OK;
}
