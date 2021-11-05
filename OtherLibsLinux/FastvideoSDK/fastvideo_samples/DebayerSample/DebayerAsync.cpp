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

#include "DebayerAsync.h"
#include "timing.hpp"
#include "checks.h"
#include "supported_files.hpp"
#include "EnumToStringSdk.h"

fastStatus_t DebayerAsync::Init(
	BaseOptions *baseOptions,
	MtResult *result,
	int threadId,
	void* specialParams
) {
	this->options = *((DebayerSampleOptions*)baseOptions);
	info = baseOptions->Info;
	mtMode = result != nullptr;


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

	fastDeviceSurfaceBufferHandle_t* bufferPtr = &srcBuffer;

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

	if (specialParams != nullptr) {
		CHECK_FAST(fastImageFilterCreate(
			&hSam,

			FAST_SAM,
			specialParams,

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
			(void*)&whiteBalanceParameter,

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

	size_t requestedMemSpace = 0;
	size_t tmp = 0;
	CHECK_FAST(fastDebayerGetAllocatedGpuMemorySize(hDebayer, &tmp));
	requestedMemSpace += tmp;
	if (hSam != nullptr) {
		CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hSam, &tmp));
		requestedMemSpace += tmp;
	}
	if (hWhiteBalance != nullptr) {
		CHECK_FAST(fastImageFiltersGetAllocatedGpuMemorySize(hWhiteBalance, &tmp));
		requestedMemSpace += tmp;
	}
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hHostToDeviceAdapter, &tmp));
	requestedMemSpace += tmp;

	const double megabyte = 1024.0 * 1024.0;
	if (mtMode && result != nullptr) {
		result->requestedMemSize = requestedMemSpace / megabyte;
	}
	else
		printf("Requested GPU memory space: %.2f MB\n\n", requestedMemSpace / megabyte);

	return FAST_OK;
}

fastStatus_t DebayerAsync::Close() const {
	CHECK_FAST(fastDebayerDestroy(hDebayer));
	if (hSam != nullptr) {
		CHECK_FAST(fastImageFiltersDestroy(hSam));
	}
	if (hWhiteBalance != nullptr) {
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

fastStatus_t DebayerAsync::Transform(
	PortableAsyncFileReader<ManagedConstFastAllocator<0>>* inImgs,
	PortableAsyncFileWriter<ManagedConstFastAllocator<1>>* outImgs,
	unsigned threadId, MtResult *result, volatile bool *terminate,
	void* specialParams
) {
	double fullTime = 0.;
	
	const hostTimer_t process_timer = hostTimerCreate();

	ManagedConstFastAllocator<1> alloc;
	// imgSurfaceSize much more bigger, than real size
	const int imgSurfaceSize = options.MaxHeight * GetPitchFromSurface(surfaceFmt, options.MaxWidth) * sizeof(unsigned char);

	size_t totalFileSize = 0;
	size_t imageCount = 0;

	double processTimeAll = 0.0;
	double releaseTimeAll = 0.0;
	double allocTimeAll = 0.0;
	double writerTimeAll = 0.0;
	double readerTimeAll = 0.0;

	while (!(*terminate)) {
		hostTimerStart(process_timer, info);
		auto inImgBatch = inImgs->ReadNextFileBatch(threadId);
		double getReaderTime = hostTimerEnd(process_timer, info);

		if (inImgBatch.IsEmpty())
			break;

		hostTimerStart(process_timer, info);
		auto outImgBatch = outImgs->GetNextWriterBatch(threadId);
		double getWriterTime = hostTimerEnd(process_timer, info);

		outImgBatch.SetFilltedItem(inImgBatch.GetFilledItem());

		double processTime = 0.0;
		double releaseTime = 0.0;
		double allocTime = 0.0;
		for (int i = 0; i < inImgBatch.GetFilledItem() && !(*terminate); i++) {
			hostTimerStart(process_timer, info);
			auto inImg = inImgBatch.At(i);
			auto outImg = outImgBatch.At(i);

			outImg->inputFileName = inImg->inputFileName;
			outImg->outputFileName = inImg->outputFileName;

			outImg->w = inImg->w / scaleFactor.x;
			outImg->wPitch = GetPitchFromSurface(surfaceFmt, outImg->w);
			outImg->h = inImg->h / scaleFactor.y;

			outImg->bitsPerChannel = inImg->bitsPerChannel;
			outImg->surfaceFmt = surfaceFmt;
			outImg->samplingFmt = inImg->samplingFmt;

			outImg->isRaw = inImg->isRaw;

			outImg->data.reset((unsigned char*)alloc.allocate(imgSurfaceSize));
			allocTime += hostTimerEnd(process_timer, info);

			if (inImg->w > options.MaxWidth ||
				inImg->h > options.MaxHeight) {
				fprintf(stderr, "Unsupported image size\n");
				continue;
			}

			hostTimerStart(process_timer);
			{
				CHECK_FAST(fastImportFromHostCopy(
					hHostToDeviceAdapter,

					inImg->data.get(),
					inImg->w,
					inImg->wPitch,
					inImg->h
				));

				if (convertTo16) {
					CHECK_FAST(fastSurfaceConverterTransform(
						hSurfaceConverterTo16,
						nullptr,

						inImg->w,
						inImg->h
					));
				}

				if (hSam != nullptr) {
					CHECK_FAST(fastImageFiltersTransform(
						hSam,
						nullptr,

						inImg->w,
						inImg->h
					));
				}

				if (hWhiteBalance != nullptr) {
					CHECK_FAST(fastImageFiltersTransform(
						hWhiteBalance,
						nullptr,

						inImg->w,
						inImg->h
					));
				}

				CHECK_FAST(fastDebayerTransform(
					hDebayer,

					options.Debayer.BayerFormat,

					inImg->w,
					inImg->h
				));

				if (convertTo16) {
					CHECK_FAST(fastSurfaceConverterTransform(
						hSurfaceConverter16to8,
						nullptr,

						outImg->w,
						outImg->h
					));
				}

				fastExportParameters_t exportParameters = { };
				exportParameters.convert = options.ConvertToBGR ? FAST_CONVERT_BGR : FAST_CONVERT_NONE;
				CHECK_FAST(fastExportToHostCopy(
					hDeviceToHostAdapter,

					outImg->data.get(),
					outImg->w,
					GetPitchFromSurface(surfaceFmt, outImg->w),
					outImg->h,

					&exportParameters
				));
			}
			totalFileSize += inImg->w * inImg->h * GetNumberOfChannelsFromSurface(surfaceFmt);

			processTime += hostTimerEnd(process_timer);

			hostTimerStart(process_timer, info);
			inImg->ReleaseBuffer();
			releaseTime += hostTimerEnd(process_timer, info);
			imageCount++;
		}

		processTimeAll += processTime;
		releaseTimeAll += releaseTime;
		allocTimeAll += allocTime;
		writerTimeAll += getWriterTime;
		readerTimeAll += getReaderTime;
	}

	outImgs->WriterFinished(threadId);

	if (mtMode) {
		fullTime = processTimeAll + releaseTimeAll + allocTimeAll + writerTimeAll + readerTimeAll;
		result->totalTime = fullTime;
		result->totalFileSize = totalFileSize;
		result->pipelineHostTime = processTimeAll;
		result->processedItem = imageCount;
		result->readerWaitTime = readerTimeAll;
		result->writerWaitTime = writerTimeAll;
		result->allocationTime = releaseTimeAll + allocTimeAll;
	}

	hostTimerDestroy(process_timer);

	return FAST_OK;
}
