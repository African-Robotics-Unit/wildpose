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
#include <thread>

#include "fastvideo_sdk.h"
#include "FastAllocator.h"

#include "barrier.hpp"

#include "Jpeg2JpegAsync.h"

#include "checks.h"

#include "helper_bytestream.hpp"

#include "Jpeg2JpegSampleOptions.h"
#include "helper_jpeg.hpp"

#include "supported_files.hpp"
#include "ResizeHelper.h"

#include "BatchedQueue.h"
#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"
#include "CollectionFastAllocator.hpp"

#include "AsyncReaderTaskEnqueuer.hpp"
#include "AsyncFileReader.hpp"
#include "AsyncFileWriter.hpp"
#include "AsyncProcessor.hpp"

#include "PreloadJpeg.hpp"
#include "MultiThreadInfoPrinter.h"
#include "AsyncReaderTaskEnqueuerFactory.hpp"

#include "AsyncProcessorFactory.hpp"

fastStatus_t RunJpeg2JpegAsync(Jpeg2JpegSampleOptions options) {
	const int fileReaderThreadCount = options.NumberOfReaderThreads;
	const int processorThreadCount = options.NumberOfThreads;
	const int fileWriterThreadCount = options.NumberOfWriterThreads;

	const int batchSize = options.BatchSize;
	unsigned queueSize = CheckAndUpdateQueueSize(options.Queue, processorThreadCount, fileReaderThreadCount);
	queueSize = CheckAndUpdateQueueSize(queueSize, fileWriterThreadCount, processorThreadCount);

	volatile bool terminateAll = false;

	fastJfifInfo_t jfifInfo = { };
	CHECK_FAST(PreloadJpeg(options.InputPath, options.IsFolder, &jfifInfo));
	
	options.MaxHeight = options.MaxHeight == 0 ? jfifInfo.height : options.MaxHeight;
	options.MaxWidth = options.MaxWidth == 0 ? jfifInfo.width : options.MaxWidth;
	options.SurfaceFmt = IdentifySurface(jfifInfo.bitsPerChannel, jfifInfo.jpegFmt == FAST_JPEG_Y ? 1 : 3);

	unsigned resizeInputMaxWidth = options.Crop.IsEnabled ? options.Crop.CropWidth : options.MaxWidth;
	unsigned resizeInputMaxHeight = options.Crop.IsEnabled ? options.Crop.CropHeight : options.MaxHeight;

	double maxScaleFactor = GetResizeMaxScaleFactor(resizeInputMaxWidth, resizeInputMaxHeight, options.Resize);
	if (!options.Resize.OutputHeightEnabled) {
		options.Resize.OutputHeight = GetResizeMaxHeight(resizeInputMaxHeight, maxScaleFactor);
	}

	if (maxScaleFactor > ResizerOptions::SCALE_FACTOR_MAX) {
		fprintf(stderr, "Incorrect image scale factor (%.3f). Max scale factor is %d\n", maxScaleFactor, ResizerOptions::SCALE_FACTOR_MAX);
		return FAST_INVALID_VALUE;
	}
	if (options.Resize.OutputWidth < FAST_MIN_SCALED_SIZE) {
		fprintf(stderr, "Image width %d is not supported (the smallest image size is %dx%d)\n", resizeInputMaxWidth, FAST_MIN_SCALED_SIZE, FAST_MIN_SCALED_SIZE);
	}

	if (options.Resize.OutputHeight < FAST_MIN_SCALED_SIZE) {
		fprintf(stderr, "Image height %d is not supported (the smallest image size is %dx%d)\n", resizeInputMaxHeight, FAST_MIN_SCALED_SIZE, FAST_MIN_SCALED_SIZE);
	}

	if (options.ImageFilter.SharpBefore != ImageFilterOptions::DisabledSharpConst) {
		printf("sharp_before parameters: r = 1.000, sigma = %.3f\n", options.ImageFilter.SharpBefore);
	}

	if (options.ImageFilter.SharpAfter != ImageFilterOptions::DisabledSharpConst) {
		printf("sharp_after parameters: r = 1.000, sigma = %.3f\n", options.ImageFilter.SharpAfter);
	}

	printf("Maximum scale factor: %.3f\n", maxScaleFactor);
	printf("Log file: %s\n", options.LogFile != NULL ? options.LogFile : "not set");

	ManagedFastAllocator<0> jfifInMgr;
	jfifInMgr.initManager(batchSize * queueSize);

	ManagedConstFastAllocator<0> jfifOutMgr;
	{
		const size_t outputPitch = GetPitchFromSurface(options.SurfaceFmt, options.Resize.OutputWidth);
		const size_t outputSize = outputPitch * options.Resize.OutputHeight;
		jfifOutMgr.initManager(batchSize * queueSize, outputSize);
	}

	CollectionFastAllocator<0> singleImgMgr;
	singleImgMgr.initManager(
		batchSize);

	auto* jpegWriter = new JpegAsyncFileWriter<ManagedConstFastAllocator<0>>();
	jpegWriter->Init(batchSize, queueSize, fileWriterThreadCount, processorThreadCount, options.Discard, &terminateAll);

	AsyncReaderTaskEnqueuer* fileEnqueuer = nullptr;
	BaseOptions* boptions = &options;

	IAsyncFileReader* reader = nullptr;
	IAsyncProcessor* processor = nullptr;


	AsyncProcessorFactory<
		Jpeg2JpegAsync<JpegAsyncFileReader<ManagedFastAllocator<0>>>,
		Jpeg2JpegAsync<JpegAsyncSingleFileReader<CollectionFastAllocator<0>>>,
		JpegAsyncFileReader<ManagedFastAllocator<0>>,
		JpegAsyncSingleFileReader<CollectionFastAllocator<0>>,
		JpegAsyncFileWriter<ManagedConstFastAllocator<0>> >
		(&fileEnqueuer, &reader, &processor,
			jpegWriter, boptions, batchSize,
			queueSize, fileReaderThreadCount,
			processorThreadCount, &terminateAll);
	
	if (fileEnqueuer != nullptr)
		fileEnqueuer->WaitAll();

	processor->WaitAll();
	reader->WaitAll();
	jpegWriter->WaitAll();

	if (!terminateAll) {
		PrintProcessorResults(processor, &options, "JPEG to JPEG");
	}
	
	delete reader;
	delete jpegWriter;
	delete processor;
	if (fileEnqueuer != nullptr)
		delete fileEnqueuer;

	jfifInMgr.freeAll();
	jfifOutMgr.freeAll();
	singleImgMgr.freeAll();

	return FAST_OK;
}
