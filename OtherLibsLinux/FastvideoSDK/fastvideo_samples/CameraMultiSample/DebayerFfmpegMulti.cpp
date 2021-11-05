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

#include "DebayerFfmpegMulti.hpp"
#include "checks.h"

fastStatus_t DebayerFfmpegMulti::Init(
	CameraMultiSampleOptions &options,
	std::unique_ptr<unsigned char, FastAllocator> &lut_0, float *matrixA_0, char *matrixB_0,
	std::unique_ptr<unsigned char, FastAllocator> &lut_1, float *matrixA_1, char *matrixB_1
) {
	options_ = options;

	CHECK_FAST(fastImportFromHostCreate(
		&hImportFromHost,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&madMuxBuffer[0]
	));

	if (options.SurfaceFmt == FAST_BGR8) {
		options.SurfaceFmt = FAST_RGB8;
	}

	int muxInput = 1;
	if (matrixA_0 != NULL || matrixB_0 != NULL) {
		fastSam_t madParameter = { 0 };
		madParameter.correctionMatrix = matrixA_0;
		madParameter.blackShiftMatrix = matrixB_0;

		CHECK_FAST(fastImageFilterCreate(
			&hSam_0,

			FAST_SAM,
			static_cast<void *>(&madParameter),

			options.MaxWidth,
			options.MaxHeight,

			madMuxBuffer[0],
			&madMuxBuffer[1]
		));
		muxInput++;
	}
	if (matrixA_1 != NULL || matrixB_1 != NULL) {
		fastSam_t madParameter = { 0 };
		madParameter.correctionMatrix = matrixA_1;
		madParameter.blackShiftMatrix = matrixB_1;

		CHECK_FAST(fastImageFilterCreate(
			&hSam_1,

			FAST_SAM,
			static_cast<void *>(&madParameter),

			options.MaxWidth,
			options.MaxHeight,

			madMuxBuffer[0],
			&madMuxBuffer[2]
		));
		muxInput++;
	}
	else
		madMuxBuffer[2] = NULL;

	CHECK_FAST(fastMuxCreate(
		&hSamMux,

		madMuxBuffer,
		muxInput,

		&srcBuffer
	));

	fastBaseColorCorrection_t colorCorrectionParameter = { 0 };
	memcpy(colorCorrectionParameter.matrix, options.BaseColorCorrection_0.BaseColorCorrection, 12 * sizeof(float));

	CHECK_FAST(fastImageFilterCreate(
		&hColorCorrection,

		FAST_BASE_COLOR_CORRECTION,
		static_cast<void *>(&colorCorrectionParameter),

		options.MaxWidth,
		options.MaxHeight,

		srcBuffer,
		&colorCorrectionBuffer
	));

	CHECK_FAST(fastDebayerCreate(
		&hDebayer,

		options.Debayer.BayerType,

		options.MaxWidth,
		options.MaxHeight,

		colorCorrectionBuffer,
		&debayerBuffer
	));

	fastLut_8_t lutParameter_0 = { 0 };
	memcpy(lutParameter_0.lut, lut_0.get(), 256 * sizeof(unsigned char));

	CHECK_FAST(fastImageFilterCreate(
		&hLut_0,

		FAST_LUT_8_8,
		static_cast<void *>(&lutParameter_0),

		options.MaxWidth,
		options.MaxHeight,

		debayerBuffer,
		&lutMuxBuffer[0]
	));

	fastLut_8_t lutParameter_1 = { 0 };
	memcpy(lutParameter_1.lut, lut_1.get(), 256 * sizeof(unsigned char));

	CHECK_FAST(fastImageFilterCreate(
		&hLut_1,

		FAST_LUT_8_8,
		static_cast<void *>(&lutParameter_1),

		options.MaxWidth,
		options.MaxHeight,

		debayerBuffer,
		&lutMuxBuffer[1]
	));

	CHECK_FAST(fastMuxCreate(
		&hLutMux,

		lutMuxBuffer,
		2,

		&lutBuffer
	));

	CHECK_FAST(fastJpegEncoderCreate(
		&hEncoder,

		options.MaxWidth,
		options.MaxHeight,

		lutBuffer
	));

	CHECK_FAST(fastMJpegAsyncWriterCreate(
		&hMjpeg,

		options.MaxWidth,
		options.MaxHeight,

		static_cast<int>(options.FFMPEG.FrameRate),
		WorkItemQueueLength,
		MaxWritersCount
	));

	fastMJpegFileDescriptor_t fileDescriptor0 = { 0 }, fileDescriptor1 = { 0 };
	{
		fileDescriptor0.fileName = options.OutputPath;
		fileDescriptor0.height = options.MaxHeight;
		fileDescriptor0.width = options.MaxWidth;
		fileDescriptor0.samplingFmt = options.JpegEncoder.SamplingFmt;

		fileDescriptor1.fileName = options.OutputPath_2;
		fileDescriptor1.height = options.MaxHeight;
		fileDescriptor1.width = options.MaxWidth;
		fileDescriptor1.samplingFmt = options.JpegEncoder.SamplingFmt;
	}

	CHECK_FAST(fastMJpegAsyncWriterOpenFile(hMjpeg, &fileDescriptor0, &FileIndex0));
	CHECK_FAST(fastMJpegAsyncWriterOpenFile(hMjpeg, &fileDescriptor1, &FileIndex1));
	if (FileIndex0 < 0 || FileIndex1 < 0) {
		fprintf(stderr, "Cannot create output file\n");
		return FAST_IO_ERROR;
	}

	size_t requestedMemSpace = 0;
	size_t tmp = 0;
	CHECK_FAST(fastJpegEncoderGetAllocatedGpuMemorySize(hEncoder, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastDebayerGetAllocatedGpuMemorySize(hDebayer, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hLut_0, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hLut_1, &tmp));
	requestedMemSpace += tmp;
	if (hSam_0 != NULL) {
		CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hSam_0, &tmp));
		requestedMemSpace += tmp;
	}
	if (hSam_1 != NULL) {
		CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hSam_1, &tmp));
		requestedMemSpace += tmp;
	}
	CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hColorCorrection, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;
	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t DebayerFfmpegMulti::StoreFrame(Image<FastAllocator> &image, int cameraId) {
	float totalTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t hostToDeviceTimer = NULL;
	fastGpuTimerHandle_t madTimer = NULL;
	fastGpuTimerHandle_t colorCorrectionTimer = NULL;
	fastGpuTimerHandle_t debayerTimer = NULL;
	fastGpuTimerHandle_t lutTimer = NULL;
	fastGpuTimerHandle_t encoderTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
		fastGpuTimerCreate(&madTimer);
		fastGpuTimerCreate(&colorCorrectionTimer);
		fastGpuTimerCreate(&debayerTimer);
		fastGpuTimerCreate(&lutTimer);
		fastGpuTimerCreate(&encoderTimer);
	}

	if (image.w > options_.MaxWidth ||
		image.h > options_.MaxHeight) {
		fprintf(stderr, "Unsupported image size\n");
		return FAST_INVALID_SIZE;
	}

	for (int i = 0; i < options_.FFMPEG.FrameRepeat; i++) {
		if (info) {
			fastGpuTimerStart(hostToDeviceTimer);
		}

		CHECK_FAST(fastImportFromHostCopy(
			hImportFromHost,

			image.data.get(),
			image.w,
			image.wPitch,
			image.h
		));

		if (info) {
			fastGpuTimerStop(hostToDeviceTimer);
			fastGpuTimerGetTime(hostToDeviceTimer, &elapsedTimeGpu);

			totalTime = elapsedTimeGpu;
			printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);
		}

		const bool IsMadRequired = (hSam_0 != NULL && cameraId == 0) || (hSam_1 != NULL && cameraId == 1);
		int madBufferIdx = 0; // default: MAD was not initialized
		if (IsMadRequired) {
			if (info) {
				fastGpuTimerStart(madTimer);
			}

			CHECK_FAST(fastImageFiltersTransform(
				cameraId == 0 ? hSam_0 : hSam_1,
				NULL,

				image.w,
				image.h
			));

			if (info) {
				fastGpuTimerStop(madTimer);
				fastGpuTimerGetTime(madTimer, &elapsedTimeGpu);

				totalTime += elapsedTimeGpu;
				printf("MAD time = %.2f ms\n", elapsedTimeGpu);
			}

			madBufferIdx = cameraId == 0 ? 1 : 2;
		}

		if (info) {
			fastGpuTimerStart(colorCorrectionTimer);
		}

		CHECK_FAST(fastMuxSelect(hSamMux, madBufferIdx));
		fastBaseColorCorrection_t colorCorrectionParameter = { 0 };
		memcpy(
			colorCorrectionParameter.matrix,
			cameraId == 0 ? options_.BaseColorCorrection_0.BaseColorCorrection : options_.BaseColorCorrection_1.BaseColorCorrection,
			12 * sizeof(float)
		);

		CHECK_FAST(fastImageFiltersTransform(
			hColorCorrection,
			&colorCorrectionParameter,

			image.w,
			image.h
		));

		if (info) {
			fastGpuTimerStop(colorCorrectionTimer);
			fastGpuTimerGetTime(colorCorrectionTimer, &elapsedTimeGpu);

			totalTime += elapsedTimeGpu;
			printf("Color correction time = %.2f ms\n", elapsedTimeGpu);

			fastGpuTimerStart(debayerTimer);
		}

		CHECK_FAST(fastDebayerTransform(
			hDebayer,

			options_.Debayer.BayerFormat,

			image.w,
			image.h
		));

		if (info) {
			fastGpuTimerStop(debayerTimer);
			fastGpuTimerGetTime(debayerTimer, &elapsedTimeGpu);

			totalTime += elapsedTimeGpu;
			printf("Debayer time = %.2f ms\n", elapsedTimeGpu);

			fastGpuTimerStart(lutTimer);
		}

		CHECK_FAST(fastImageFiltersTransform(
			cameraId == 0 ? hLut_0 : hLut_1,
			NULL,

			image.w,
			image.h
		));

		if (info) {
			fastGpuTimerStop(lutTimer);
			fastGpuTimerGetTime(lutTimer, &elapsedTimeGpu);

			totalTime += elapsedTimeGpu;
			printf("Lut time = %.2f ms\n", elapsedTimeGpu);

			fastGpuTimerStart(encoderTimer);
		}

		fastJfifInfo_t *jfifInfo = NULL;
		const fastStatus_t ret = fastMJpegAsyncWriterGetJfifInfo(hMjpeg, &jfifInfo);
		if (ret != FAST_OK) {
			fastMJpegError_t *error = NULL;
			CHECK_FAST(fastMJpegAsyncWriterGetErrorStatus(hMjpeg, &error));
			fprintf(stderr, "Motion JPEG writer error: %s\n", error->fileName);
			Close();
			return FAST_INTERNAL_ERROR;
		}

		jfifInfo->width = image.w;
		jfifInfo->height = image.h;
		jfifInfo->restartInterval = options_.JpegEncoder.RestartInterval;
		jfifInfo->jpegFmt = options_.JpegEncoder.SamplingFmt;

		CHECK_FAST(fastMuxSelect(hLutMux, cameraId));
		CHECK_FAST(fastJpegEncode(
			hEncoder,

			options_.JpegEncoder.Quality,
			jfifInfo
		));

		if (info) {
			fastGpuTimerStop(encoderTimer);
			fastGpuTimerGetTime(encoderTimer, &elapsedTimeGpu);

			printf("Encode time (includes device-to-host transfer) = %.2f ms\n", elapsedTimeGpu);
			totalTime += elapsedTimeGpu;
		}

		CHECK_FAST(fastMJpegAsyncWriteFrame(hMjpeg, jfifInfo, cameraId == 0 ? FileIndex0 : FileIndex1));
	}
	

	if (info) {
		fastGpuTimerDestroy(hostToDeviceTimer);
		fastGpuTimerDestroy(debayerTimer);
		fastGpuTimerDestroy(lutTimer);
		fastGpuTimerDestroy(encoderTimer);
	}

	return FAST_OK;
}

fastStatus_t DebayerFfmpegMulti::Close(void) const {
	CHECK_FAST(fastDebayerDestroy(hDebayer));
	CHECK_FAST(fastImageFiltersDestroy(hLut_0));
	CHECK_FAST(fastImageFiltersDestroy(hLut_1));
	if (hSam_0 != NULL) {
		CHECK_FAST(fastImageFiltersDestroy(hSam_0));
	}
	if (hSam_1 != NULL) {
		CHECK_FAST(fastImageFiltersDestroy(hSam_1));
	}
	CHECK_FAST(fastImageFiltersDestroy(hColorCorrection));
	CHECK_FAST(fastJpegEncoderDestroy(hEncoder));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));

	CHECK_FAST(fastMuxDestroy(hSamMux));
	CHECK_FAST(fastMuxDestroy(hLutMux));

	CHECK_FAST(fastMJpegAsyncWriterClose(hMjpeg));

	return FAST_OK;
}
