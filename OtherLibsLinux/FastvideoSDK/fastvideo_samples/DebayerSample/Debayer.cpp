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
#include <vector_functions.h>

#include "Debayer.h"
#include "supported_files.hpp"
#include "checks.h"
#include "BaseOptions.h"
#include "SurfaceTraits.hpp"

fastStatus_t Debayer::Init(DebayerSampleOptions &options, float *matrixA, char *matrixB, MtResult *result) {
	this->options = options;
	switch (options.Debayer.BayerType) {
		case FAST_MG:
		case FAST_BINNING_2x2:
		{
			if ((options.MaxWidth & 1) != 0) {
				fprintf(stderr, "Unsupported image size\n");
				return FAST_INVALID_SIZE;
			}
			break;
		}
		case FAST_BINNING_4x4:
		{
			if ((options.MaxWidth & 2) != 0) {
				fprintf(stderr, "Unsupported image size\n");
				return FAST_INVALID_SIZE;
			}
			break;
		}
		case FAST_BINNING_8x8:
		{
			if ((options.MaxWidth & 3) != 0) {
				fprintf(stderr, "Unsupported image size\n");
				return FAST_INVALID_SIZE;
			}
			break;
		}
	}

	convertTo16 = false;
	if (options.Debayer.BayerType == FAST_MG && options.SurfaceFmt == FAST_I8) {
		printf("Debayer MG does not support 8 bits image. Image has been converted to 16 bits.");
		convertTo16 = true;
	}
	if (
		(options.Debayer.BayerType == FAST_BINNING_2x2 ||
			options.Debayer.BayerType == FAST_BINNING_4x4 ||
			options.Debayer.BayerType == FAST_BINNING_8x8)
		&& options.SurfaceFmt == FAST_I12
	) {
		printf("Debayer Binning does not support 12 bits image. Image has been converted to 16 bits.");
		convertTo16 = true;
	}

	CHECK_FAST(fastImportFromHostCreate(
		&hHostToDeviceAdapter,

		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,

		&srcBuffer
	));

	fastDeviceSurfaceBufferHandle_t *bufferPtr = &srcBuffer;

	if (convertTo16) {
		fastBitDepthConverter_t bitDepthParam = { 0 };
		{
			bitDepthParam.isOverrideSourceBitsPerChannel = false;
			bitDepthParam.targetBitsPerChannel = 16;
		}
		fastSurfaceConverterCreate(
			&hSurfaceConverterTo16,

			FAST_BIT_DEPTH,
			static_cast<void*>(&bitDepthParam),

			options.MaxWidth,
			options.MaxHeight,

			srcBuffer,
			&bufferTo16
		);
		bufferPtr = &bufferTo16;
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

			*bufferPtr,
			&madBuffer
		));

		bufferPtr = &madBuffer;
	}

	if (options.WhiteBalance.R != 1.0f || options.WhiteBalance.G1 != 1.0f || options.WhiteBalance.G2 != 1.0f || options.WhiteBalance.B != 1.0f) {
		fastWhiteBalance_t whiteBalanceParameter = { 0 };
		whiteBalanceParameter.bayerPattern = options.Debayer.BayerFormat;
		whiteBalanceParameter.R = options.WhiteBalance.R;
		whiteBalanceParameter.G1 = options.WhiteBalance.G1;
		whiteBalanceParameter.G2 = options.WhiteBalance.G2;
		whiteBalanceParameter.B = options.WhiteBalance.B;

		CHECK_FAST(fastImageFilterCreate(
			&hWhiteBalance,

			FAST_WHITE_BALANCE,
			(void *)&whiteBalanceParameter,

			options.MaxWidth,
			options.MaxHeight,

			*bufferPtr,
			&whiteBalanceBuffer
		));

		bufferPtr = &whiteBalanceBuffer;
	}

	CHECK_FAST(fastDebayerCreate(
		&hDebayer,

		options.Debayer.BayerType,

		options.MaxWidth,
		options.MaxHeight,

		*bufferPtr,
		&debayerBuffer
	));

	bufferPtr = &debayerBuffer;
	switch (options.Debayer.BayerType) {
		case FAST_BINNING_2x2:
		{
			scaleFactor = make_uint2(2, 2);
			break;
		}
		case FAST_BINNING_4x4:
		{
			scaleFactor = make_uint2(4, 4);
			break;
		}
		case FAST_BINNING_8x8:
		{
			scaleFactor = make_uint2(8, 8);
			break;
		}
		default:
		{
			scaleFactor = make_uint2(1, 1);
			break;
		}
	}

	if (convertTo16) {
		fastBitDepthConverter_t bitDepthParam = { 0 };

		bitDepthParam.isOverrideSourceBitsPerChannel = false;
		bitDepthParam.targetBitsPerChannel = GetBitsPerChannelFromSurface(options.SurfaceFmt);

		fastSurfaceConverterCreate(
			&hSurfaceConverter16to8,

			FAST_BIT_DEPTH,
			static_cast<void*>(&bitDepthParam),

			options.MaxWidth / scaleFactor.x,
			options.MaxHeight / scaleFactor.y,

			*bufferPtr,
			&buffer16to8
		);
		bufferPtr = &buffer16to8;
	}

	CHECK_FAST(fastExportToHostCreate(
		&hDeviceToHostAdapter,
		&surfaceFmt,
		*bufferPtr
	));

	FastAllocator allocator;
	unsigned pitch = GetPitchFromSurface(surfaceFmt, options.MaxWidth / scaleFactor.x);
	CHECK_FAST_ALLOCATION(h_Result.reset((unsigned char *)allocator.allocate(pitch * (options.MaxHeight / scaleFactor.y) * sizeof(unsigned char))));

	size_t requestedMemSpace = 0;
	size_t tmp = 0;
	CHECK_FAST(fastDebayerGetAllocatedGpuMemorySize(hDebayer, &tmp));
	requestedMemSpace += tmp;
	if (hSam != NULL) {
		CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hSam, &tmp));
		requestedMemSpace += tmp;
	}
	if (hWhiteBalance != NULL) {
		CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hWhiteBalance, &tmp));
		requestedMemSpace += tmp;
	}
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hHostToDeviceAdapter, &tmp));
	requestedMemSpace += tmp;
	const double megabyte = 1024.0 * 1024.0;
	if (mtMode && result != nullptr) {
		result->requestedMemSize = requestedMemSpace / megabyte;
	} else {
		printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / megabyte);
	}

	return FAST_OK;
}

fastStatus_t Debayer::Transform(std::list<Image<FastAllocator> > &image, int threadId, MtResult *result) {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	float processAvgTimeGpu = 0.;
	float processMinTimeGpu = 99999.;
	float processMaxTimeGpu = -99999.;

	fastGpuTimerHandle_t hostToDeviceTimer = NULL;
	fastGpuTimerHandle_t debayerTimer = NULL;
	fastGpuTimerHandle_t madTimer = NULL;
	fastGpuTimerHandle_t colorCorrectionTimer = NULL;
	fastGpuTimerHandle_t deviceToHostTimer = NULL;
	fastGpuTimerHandle_t converterTimer = NULL;

	hostTimer_t timer = hostTimerCreate();
	fastGpuTimerCreate(&debayerTimer);

	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
		fastGpuTimerCreate(&madTimer);
		fastGpuTimerCreate(&colorCorrectionTimer);
		fastGpuTimerCreate(&deviceToHostTimer);
		fastGpuTimerCreate(&converterTimer);
	}

	double totalFileSize = 0.0;
	for (auto i = image.begin(); i != image.end(); ++i) {
		Image<FastAllocator> &img = *i;

		if (threadId == 0) {
			printf("Input image: %s\nImage size: %dx%d pixels, %d bits\n\n", img.inputFileName.c_str(), img.w, img.h, img.bitsPerChannel);
		}

		if (img.w > options.MaxWidth ||
			img.h > options.MaxHeight) {
			fprintf(stderr, "Unsupported image size\n");
			continue;
		}

		if (options.Debayer.BayerType == FAST_MG) {
			if ((img.w & 1) != 0) {
				fprintf(stderr, "Unsupported image size\n");
				continue;
			}
		}

		hostTimerStart(timer);

		for (int j = 0; j < options.RepeatCount; j++) {
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

				fullTime += elapsedTimeGpu;
				if (!mtMode) {
					printf("Host-to-device transfer = %.2f ms\n", elapsedTimeGpu);
				}
			}

			if (convertTo16) {
				if (info) {
					fastGpuTimerStart(converterTimer);
				}
				CHECK_FAST(fastSurfaceConverterTransform(
					hSurfaceConverterTo16,
					NULL,

					img.w,
					img.h
				));

				if (info) {
					fastGpuTimerStop(converterTimer);
					fastGpuTimerGetTime(converterTimer, &elapsedTimeGpu);

					fullTime += elapsedTimeGpu;
					if (!mtMode) {
						printf("Converter to 16 time = %.2f ms\n", elapsedTimeGpu);
					}
				}
			}

			if (hSam != NULL) {
				if (info) {
					fastGpuTimerStart(madTimer);
				}

				CHECK_FAST(fastImageFiltersTransform(
					hSam,
					NULL,

					img.w,
					img.h
				));

				if (info) {
					fastGpuTimerStop(madTimer);
					fastGpuTimerGetTime(madTimer, &elapsedTimeGpu);

					fullTime += elapsedTimeGpu;
					if (!mtMode) {
						printf("MAD time = %.2f ms\n", elapsedTimeGpu);
					}
				}
			}

			if (hWhiteBalance != NULL) {
				if (info) {
					fastGpuTimerStart(colorCorrectionTimer);
				}

				CHECK_FAST(fastImageFiltersTransform(
					hWhiteBalance,
					NULL,

					img.w,
					img.h
				));

				if (info) {
					fastGpuTimerStop(colorCorrectionTimer);
					fastGpuTimerGetTime(colorCorrectionTimer, &elapsedTimeGpu);

					fullTime += elapsedTimeGpu;
					if (!mtMode && options.RepeatCount == 1) {
						printf("Color correction time = %.2f ms\n", elapsedTimeGpu);
					}
				}
			}

			if (!mtMode) {
				fastGpuTimerStart(debayerTimer);
			}

			CHECK_FAST(fastDebayerTransform(
				hDebayer,

				options.Debayer.BayerFormat,

				img.w,
				img.h
			));

			if (!mtMode) {
				fastGpuTimerStop(debayerTimer);
				fastGpuTimerGetTime(debayerTimer, &elapsedTimeGpu);
				fullTime += elapsedTimeGpu;
				processMaxTimeGpu = std::max(processMaxTimeGpu, elapsedTimeGpu);
				processMinTimeGpu = std::min(processMinTimeGpu, elapsedTimeGpu);
				processAvgTimeGpu += elapsedTimeGpu;
			}

			const uint2 dstSize = make_uint2(
				img.w / scaleFactor.x,
				img.h / scaleFactor.y
			);

			if (info) {
				if (!mtMode) {
					printf("Effective debayer performance = %.2f Gpixel/s (%.2f ms)\n", double(img.h * img.w) / elapsedTimeGpu * 1E-6, elapsedTimeGpu);
				}
				fastGpuTimerStart(deviceToHostTimer);
			}

			if (convertTo16) {
				if (info) {
					fastGpuTimerStart(converterTimer);
				}

				CHECK_FAST(fastSurfaceConverterTransform(
					hSurfaceConverter16to8,
					NULL,

					dstSize.x,
					dstSize.y
				));

				if (info) {
					fastGpuTimerStop(converterTimer);
					fastGpuTimerGetTime(converterTimer, &elapsedTimeGpu);

					fullTime += elapsedTimeGpu;
					if (!mtMode) {
						printf("Converter 16 to %d time = %.2f ms\n", GetBitsPerChannelFromSurface(options.SurfaceFmt), elapsedTimeGpu);
					}
				}
			}

			fastExportParameters_t exportParameters = { };
			exportParameters.convert = options.ConvertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
			CHECK_FAST(fastExportToHostCopy(
				hDeviceToHostAdapter,

				h_Result.get(),
				dstSize.x,
				GetPitchFromSurface(surfaceFmt, dstSize.x),
				dstSize.y,

				&exportParameters
			));

			if (info) {
				fastGpuTimerStop(deviceToHostTimer);
				fastGpuTimerGetTime(deviceToHostTimer, &elapsedTimeGpu);

				fullTime += elapsedTimeGpu;
				if (!mtMode && options.RepeatCount == 1) {
					printf("Device-to-host transfer = %.2f ms\n", elapsedTimeGpu);
				}
			}
		}

		if (mtMode) {
			const double elapsedTime = hostTimerEnd(timer);
			fullTime += static_cast<float>(elapsedTime*1000.0f);
		}

		const double inSize = img.h * img.w;
		totalFileSize += options.RepeatCount * inSize;

		printf("Output image: %s\n\n", img.outputFileName.c_str());

		CHECK_FAST_SAVE_FILE(fvSaveImageToFile(
			(char *)img.outputFileName.c_str(),
			h_Result,
			surfaceFmt,
			img.h / scaleFactor.y,
			img.w / scaleFactor.x,
			GetPitchFromSurface(surfaceFmt, img.w / scaleFactor.x),
			false
		));
	}

	const unsigned imageCount = static_cast<unsigned>(image.size() * options.RepeatCount);

	if (mtMode) {
		result->totalTime = fullTime;
		result->totalFileSize = totalFileSize;
	} else {
		if (info) {
			printf("Processing time on GPU for %d images including all transfers = %.2f ms; %.0f MPixel/s;  %.0f FPS\n",
				imageCount, fullTime, totalFileSize / (fullTime * 1000.0), 1000.0 * imageCount / fullTime);

			printf("Processing time on GPU for %d images excluding host-to-device and device-to-host transfers: average = %.3f ms (%.0f MPixel/s;  %.0f FPS), min = %.3f ms, max = %.3f ms\n",
				imageCount, processAvgTimeGpu / imageCount, totalFileSize / (processAvgTimeGpu * 1000.0), 1000.0 * imageCount / processAvgTimeGpu,
				processMinTimeGpu, processMaxTimeGpu);
		} else
			printf("Processing time on GPU for %d images excluding host-to-device and device-to-host transfers = %.2f ms; %.0f MPixel/s;  %.0f FPS\n",
				imageCount, fullTime, totalFileSize / (fullTime * 1000.0), 1000.0 * imageCount / fullTime);
	}


	fastGpuTimerDestroy(debayerTimer);
	hostTimerDestroy(timer);

	if (info) {
		fastGpuTimerDestroy(hostToDeviceTimer);
		fastGpuTimerDestroy(madTimer);
		fastGpuTimerDestroy(colorCorrectionTimer);
		fastGpuTimerDestroy(deviceToHostTimer);
		fastGpuTimerDestroy(converterTimer);
	}

	return FAST_OK;
}

fastStatus_t Debayer::Close(void) const {
	CHECK_FAST(fastDebayerDestroy(hDebayer));
	if (hSam != NULL) {
		CHECK_FAST(fastImageFiltersDestroy(hSam));
	}
	if (hWhiteBalance != NULL) {
		CHECK_FAST(fastImageFiltersDestroy(hWhiteBalance));
	}
	CHECK_FAST(fastImportFromHostDestroy(hHostToDeviceAdapter));
	CHECK_FAST(fastExportToHostDestroy(hDeviceToHostAdapter));

	if (convertTo16) {
		fastSurfaceConverterDestroy(hSurfaceConverter16to8);
		fastSurfaceConverterDestroy(hSurfaceConverterTo16);
	}
	return FAST_OK;
}
