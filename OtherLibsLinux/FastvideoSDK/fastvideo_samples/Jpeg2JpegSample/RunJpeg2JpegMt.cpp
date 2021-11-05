/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <thread>

#include "Jpeg2Jpeg.h"
#include "Jpeg2JpegSampleOptions.h"

#include "checks.h"
#include "helper_jpeg.hpp"
#include "helper_bytestream.hpp"
#include "supported_files.hpp"
#include "FastAllocator.h"

#include "barrier.hpp"
#include "MultiThreadIndo.hpp"
#include "cuda_runtime.h"
#include "ResizeHelper.h"

volatile bool failedJpeg2Jpeg;
barrier *barrierJpeg2Jpeg;

struct Jpeg2JpegThreadParam {
public:
	Jpeg2JpegSampleOptions options;
	std::list< Bytestream<FastAllocator> > inputImg;
	double maxScaleFactor;
	MtResult result;
	int threadId;
	int sleepTime;
};

fastStatus_t CallJpeg2Jpeg(
	Jpeg2JpegSampleOptions &options,
	std::list<Bytestream<FastAllocator> > &inputImg,
	double maxScaleFactor,
	MtResult *result, int threadId, int sleepTime
) {
	Jpeg2Jpeg hJpeg2Jpeg(options.Info, true);

	fastStatus_t ret = hJpeg2Jpeg.Init(options, maxScaleFactor, result);
	barrierJpeg2Jpeg->wait();
	std::chrono::milliseconds timespan(sleepTime);
	std::this_thread::sleep_for(timespan);

	if (ret != FAST_OK) {
		if (ret == FAST_INSUFFICIENT_DEVICE_MEMORY)
			fprintf(stderr, "Insufficient device memory in thread %d.\n", threadId);

		failedJpeg2Jpeg = true;
		result->isCreated = false;
		return ret;
	}

	result->isCreated = true;
	CHECK_FAST(hJpeg2Jpeg.Resize(inputImg, threadId, result));
	CHECK_FAST(hJpeg2Jpeg.Close());

	return FAST_OK;
}

void *Jpeg2JpegThreadFunc(Jpeg2JpegThreadParam *data) {
	data->result.lastStatus = CallJpeg2Jpeg(data->options, data->inputImg, data->maxScaleFactor, &data->result, data->threadId, data->sleepTime);
	return NULL;
}

fastStatus_t RunJpeg2JpegMt(Jpeg2JpegSampleOptions &options) {
	std::list< Bytestream<FastAllocator> > inputImgs;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadBytestreams(options.InputPath, inputImgs, options.Info));
		int idx = 0;
		for (auto i = inputImgs.begin(); i != inputImgs.end(); i++, idx++) {
			i->outputFileName = generateOutputFileName(options.OutputPath, idx);
		}

	} else {
		Bytestream<FastAllocator> inputImg;
		CHECK_FAST(fvLoadBytestream(std::string(options.InputPath), inputImg, options.Info));
		inputImgs.push_back(inputImg);

		(--inputImgs.end())->outputFileName = std::string(options.OutputPath);
	}

	if (options.Crop.IsEnabled) {
		printf("Crop left border coords: %dx%d pixels\n", options.Crop.CropLeftTopCoordsX, options.Crop.CropLeftTopCoordsY);
		printf("Cropped image size: %dx%d pixels\n", options.Crop.CropWidth, options.Crop.CropHeight);

		if (options.Resize.OutputWidth > options.Crop.CropWidth) {
			fprintf(stderr, "Output width (%d) is bigger than cropped (%d)\n", options.Resize.OutputWidth, options.Crop.CropWidth);
			return FAST_INVALID_SIZE;
		}
	}

	fastJfifInfo_t jfifInfo = { };
	{
		jfifInfo.h_Bytestream = NULL;
		jfifInfo.exifSections = NULL;
		jfifInfo.exifSectionsCount = 0;

		jfifInfo.bytestreamSize = (*inputImgs.begin()).size;
		CHECK_FAST(fastMalloc((void **)&jfifInfo.h_Bytestream, jfifInfo.bytestreamSize));

		CHECK_FAST(fastJfifLoadFromMemory((*inputImgs.begin()).data.get(), (*inputImgs.begin()).size, &jfifInfo));

		if (jfifInfo.h_Bytestream != NULL) {
			CHECK_FAST_DEALLOCATION(fastFree(jfifInfo.h_Bytestream));
		}
		
		for (unsigned i = 0; i < jfifInfo.exifSectionsCount; i++) {
			free(jfifInfo.exifSections[i].exifData);
		}

		if (jfifInfo.exifSections != NULL) {
			free(jfifInfo.exifSections);
		}
	}
	options.SurfaceFmt = jfifInfo.jpegFmt == FAST_JPEG_Y ? FAST_I8 : FAST_RGB8;
	options.JpegEncoder.SamplingFmt = jfifInfo.jpegFmt == FAST_JPEG_Y ? FAST_JPEG_Y : options.JpegEncoder.SamplingFmt;

	if (!options.IsFolder) {
		options.MaxHeight = options.MaxHeight == 0 ? jfifInfo.height : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? jfifInfo.width : options.MaxWidth;
	}

	unsigned resizeInputMaxWidth = options.Crop.IsEnabled ? options.Crop.CropWidth : options.MaxWidth;
	unsigned resizeInputMaxHeight = options.Crop.IsEnabled ? options.Crop.CropHeight : options.MaxHeight;

	double maxScaleFactor = GetResizeMaxScaleFactor(resizeInputMaxWidth, resizeInputMaxHeight, options.Resize);
	if (!options.Resize.OutputHeightEnabled) {
		options.Resize.OutputHeight = GetResizeMaxHeight(resizeInputMaxHeight, maxScaleFactor);
	}

	if ( maxScaleFactor > ResizerOptions::SCALE_FACTOR_MAX) {
		fprintf(stderr, "Incorrect image scale factor (%.3f). Max scale factor is %d\n", maxScaleFactor, ResizerOptions::SCALE_FACTOR_MAX);
		return FAST_INVALID_SIZE;
	}

	if (options.ImageFilter.SharpBefore != ImageFilterOptions::DisabledSharpConst) {
		printf("sharp_before parameters: r = 1.000, sigma = %.3f\n", options.ImageFilter.SharpBefore);
	}

	if (options.ImageFilter.SharpAfter != ImageFilterOptions::DisabledSharpConst) {
		printf("sharp_after parameters: r = 1.000, sigma = %.3f\n", options.ImageFilter.SharpAfter);
	}

	printf("Maximum scale factor: %.3f\n", maxScaleFactor);
	printf("Log file: %s\n", options.LogFile != nullptr ? options.LogFile : "not set");

	const auto params = new Jpeg2JpegThreadParam[options.NumberOfThreads];
	for (int i = 0; i < options.NumberOfThreads; i++) {
		CreateListForThread(inputImgs, params[i].inputImg, i);
		params[i].options = options;
		params[i].threadId = i;
		params[i].maxScaleFactor = maxScaleFactor;
		params[i].result.isCreated = false;
		params[i].sleepTime = i * 2;
	}

	failedJpeg2Jpeg = false;
	barrierJpeg2Jpeg = new barrier(options.NumberOfThreads);

	const auto threads = new std::thread[options.NumberOfThreads];
	for (int i = 0; i < options.NumberOfThreads; i++) {
		threads[i] = std::thread(Jpeg2JpegThreadFunc, &params[i]);
	}

	for (int i = 0; i < options.NumberOfThreads; i++) {
		threads[i].join();
		params[i].inputImg.clear();
	}

	if (!failedJpeg2Jpeg) {
		double requestedMemSize = 0., totalTime = 0.;
		for (int i = 0; i < options.NumberOfThreads; i++) {
			requestedMemSize += params[i].result.requestedMemSize;
			totalTime = std::max(totalTime, params[i].result.totalTime);
		}
		const int imageCount = static_cast<int>(inputImgs.size()) * options.RepeatCount * options.NumberOfThreads;
		printf("Requested GPU memory size: %.2lf MB\n", requestedMemSize);
	
		printf("Processing time on CPU including all transfers for %d images (%d images per thread x %d threads) = %.1f ms; %.0f FPS;\n",
			imageCount, imageCount / options.NumberOfThreads, options.NumberOfThreads, totalTime,
			imageCount * 1000.0 / totalTime
		);
	}
	inputImgs.clear();

	delete[] params;
	delete[] threads;
	delete barrierJpeg2Jpeg;

	cudaDeviceReset();
	return FAST_OK;
}
