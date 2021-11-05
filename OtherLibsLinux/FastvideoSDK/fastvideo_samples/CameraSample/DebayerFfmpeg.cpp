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
#include <cuda_runtime.h>

#include "DebayerFfmpeg.hpp"
#include "checks.h"

fastStatus_t DebayerFfmpeg::Init(CameraSampleOptions &options, std::unique_ptr<unsigned char, FastAllocator> &lut, float *matrixA, char *matrixB) {
	quality = options.JpegEncoder.Quality;
	restartInterval = options.JpegEncoder.RestartInterval;
	jpegFmt = options.JpegEncoder.SamplingFmt;
	surfaceFmt = FAST_RGB8;
	bayer_pattern_ = options.Debayer.BayerFormat;

	frameRepeat = options.FFMPEG.FrameRepeat;
	frameRate = static_cast<int>(options.FFMPEG.FrameRate);

	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;

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

	if (matrixA != NULL || matrixB != NULL) {
		fastSam_t madParameter = { 0 };
		madParameter.correctionMatrix = matrixA;
		madParameter.blackShiftMatrix = matrixB;

		CHECK_FAST(fastImageFilterCreate(
			&hSam,

			FAST_SAM,
			(void *)&madParameter,

			options.MaxWidth,
			options.MaxHeight,

			srcBuffer,
			&madBuffer
			));
	}

	fastBaseColorCorrection_t colorCorrectionParameter = { 0 };
	memcpy(colorCorrectionParameter.matrix, options.BaseColorCorrection.BaseColorCorrection, 12 * sizeof(float));

	CHECK_FAST(fastImageFilterCreate(
		&hColorCorrection,

		FAST_BASE_COLOR_CORRECTION,
		(void *)&colorCorrectionParameter,

		options.MaxWidth,
		options.MaxHeight,

		hSam != NULL ? madBuffer : srcBuffer,
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

	fastLut_8_t lutParameter = { 0 };
	memcpy(lutParameter.lut, lut.get(), 256 * sizeof(unsigned char));

	CHECK_FAST(fastImageFilterCreate(
		&hLut,

		FAST_LUT_8_8,
		(void *)&lutParameter,

		options.MaxWidth,
		options.MaxHeight,

		debayerBuffer,
		&lutBuffer
		));

	CHECK_FAST(fastJpegEncoderCreate(
		&hEncoder,

		options.MaxWidth,
		options.MaxHeight,

		lutBuffer
		));

	CHECK_FAST(fastExportToDeviceCreate(
		&hExportToDevice,

		&surfaceFmt,
		lutBuffer
		));

	CHECK_FAST(fastMJpegAsyncWriterCreate(
		&hMjpeg,

		options.MaxWidth,
		options.MaxHeight,

		frameRate,
		WorkItemQueueLength,
		MaxWritersCount
		));

	fastMJpegFileDescriptor_t fileDescriptor = { 0 };
	fileDescriptor.fileName = options.OutputPath;
	fileDescriptor.height = options.MaxHeight;
	fileDescriptor.width = options.MaxWidth;
	fileDescriptor.samplingFmt = options.JpegEncoder.SamplingFmt;

	CHECK_FAST(fastMJpegAsyncWriterOpenFile(hMjpeg, &fileDescriptor, &FileIndex));
	if (FileIndex < 0) {
		fprintf(stderr, "Cannot create output file\n");
		return FAST_IO_ERROR;
	}

	unsigned maxPitch = 3 * (((options.MaxWidth + FAST_ALIGNMENT - 1) / FAST_ALIGNMENT) * FAST_ALIGNMENT);
	unsigned bufferSize = maxPitch * options.MaxHeight * sizeof(unsigned char);

	CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;
	CHECK_FAST(fastJpegEncoderGetAllocatedGpuMemorySize(hEncoder, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastDebayerGetAllocatedGpuMemorySize(hDebayer, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hLut, &tmp));
	requestedMemSpace += tmp;
	if (hSam != NULL) {
		CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hSam, &tmp));
		requestedMemSpace += tmp;
	}
	CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hColorCorrection, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;
	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t DebayerFfmpeg::StoreFrame(Image<FastAllocator> &image) {
	float totalTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t hostToDeviceTimer = NULL;
	fastGpuTimerHandle_t madTimer = NULL;
	fastGpuTimerHandle_t colorCorrectionTimer = NULL;
	fastGpuTimerHandle_t debayerTimer = NULL;
	fastGpuTimerHandle_t lutTimer = NULL;
	fastGpuTimerHandle_t encoderTimer = NULL;
	fastGpuTimerHandle_t deviceToSurfaceTimer = NULL;

	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
		fastGpuTimerCreate(&madTimer);
		fastGpuTimerCreate(&colorCorrectionTimer);
		fastGpuTimerCreate(&debayerTimer);
		fastGpuTimerCreate(&lutTimer);
		fastGpuTimerCreate(&encoderTimer);
		fastGpuTimerCreate(&deviceToSurfaceTimer);
	}

	if (image.w > maxWidth ||
		image.h > maxHeight) {
		fprintf(stderr, "Unsupported image size\n");
		return FAST_INVALID_SIZE;
	}

	for (int i = 0; i < frameRepeat; i++) {
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

		if (hSam != NULL) {
			if (info) {
				fastGpuTimerStart(madTimer);
			}

			CHECK_FAST(fastImageFiltersTransform(
				hSam,
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
		}

		if (info) {
			fastGpuTimerStart(colorCorrectionTimer);
		}

		CHECK_FAST(fastImageFiltersTransform(
			hColorCorrection,
			NULL,

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

			bayer_pattern_,

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
			hLut,
			NULL,

			image.w,
			image.h
		));

		if (info) {
			fastGpuTimerStop(lutTimer);
			fastGpuTimerGetTime(lutTimer, &elapsedTimeGpu);

			totalTime += elapsedTimeGpu;
			printf("Lut time = %.2f ms\n", elapsedTimeGpu);

			fastGpuTimerStart(deviceToSurfaceTimer);
		}

		CHECK_FAST(fastExportToDeviceCopy(
			hExportToDevice,

			d_buffer,
			image.w,
			image.wPitch * 3 * sizeof(char),
			image.h,

			NULL
		));

		if (info) {
			fastGpuTimerStop(deviceToSurfaceTimer);
			fastGpuTimerGetTime(deviceToSurfaceTimer, &elapsedTimeGpu);

			totalTime += elapsedTimeGpu;
			printf("Device to surface time = %.2f ms\n", elapsedTimeGpu);

			fastGpuTimerStart(encoderTimer);
		}

		fastJfifInfo_t *jfifInfo = NULL;
		if (fastMJpegAsyncWriterGetJfifInfo(hMjpeg, &jfifInfo) != FAST_OK) {
			fastMJpegError_t *error = NULL;
			CHECK_FAST(fastMJpegAsyncWriterGetErrorStatus(hMjpeg, &error));
			fprintf(stderr, "Motion JPEG writer error: %s\n", error->fileName);
			Close();
			return FAST_INTERNAL_ERROR;
		}

		jfifInfo->width = image.w;
		jfifInfo->height = image.h;
		jfifInfo->restartInterval = restartInterval;
		jfifInfo->jpegFmt = jpegFmt;

		CHECK_FAST(fastJpegEncode(
			hEncoder,

			quality,
			jfifInfo
			));

		if (info) {
			fastGpuTimerStop(encoderTimer);
			fastGpuTimerGetTime(encoderTimer, &elapsedTimeGpu);

			printf("Encode time (includes device-to-host transfer) = %.2f ms\n", elapsedTimeGpu);
			totalTime += elapsedTimeGpu;
		}

		CHECK_FAST(fastMJpegAsyncWriteFrame(hMjpeg, jfifInfo, FileIndex));
	}

	if (info) {
		fastGpuTimerDestroy(hostToDeviceTimer);
		fastGpuTimerDestroy(debayerTimer);
		fastGpuTimerDestroy(lutTimer);
		fastGpuTimerDestroy(encoderTimer);
	}

	return FAST_OK;
}

fastStatus_t DebayerFfmpeg::Close() const {
	CHECK_FAST(fastDebayerDestroy(hDebayer));
	CHECK_FAST(fastImageFiltersDestroy(hLut));
	if (hSam != NULL) {
		CHECK_FAST(fastImageFiltersDestroy(hSam));
	}
	CHECK_FAST(fastImageFiltersDestroy(hColorCorrection));
	CHECK_FAST(fastJpegEncoderDestroy(hEncoder));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));
	CHECK_FAST(fastExportToDeviceDestroy(hExportToDevice));

	CHECK_FAST(fastMJpegAsyncWriterClose(hMjpeg));

	CHECK_CUDA(cudaFree(d_buffer));

	return FAST_OK;
}

void *DebayerFfmpeg::GetDevicePtr() {
	return d_buffer;
}
