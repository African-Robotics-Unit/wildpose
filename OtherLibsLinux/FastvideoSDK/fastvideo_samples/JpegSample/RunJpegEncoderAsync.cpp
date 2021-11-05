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

#include "EncoderAsync.h"

#include "checks.h"

#include "helper_bytestream.hpp"

#include "JpegEncoderSampleOptions.h"
#include "helper_jpeg.hpp"
#include "helper_quant_table.hpp"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"

#include "BatchedQueue.h"
#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"
#include "CollectionFastAllocator.hpp"

#include "AsyncReaderTaskEnqueuer.hpp"
#include "AsyncFileReader.hpp"
#include "AsyncFileWriter.hpp"
#include "AsyncProcessor.hpp"

#include "PreloadUncompressedImage.hpp"
#include "MultiThreadInfoPrinter.h"
#include "AsyncProcessorFactory.hpp"


fastStatus_t RunJpegEncoderAsync(JpegEncoderSampleOptions &options) {
	const int fileReaderThreadCount = options.NumberOfReaderThreads;
	const int processorThreadCount = options.NumberOfThreads;
	const int fileWriterThreadCount = options.NumberOfWriterThreads;

	const int batchSize = options.BatchSize;
	unsigned queueSize = CheckAndUpdateQueueSize(options.Queue, processorThreadCount, fileReaderThreadCount);
	queueSize = CheckAndUpdateQueueSize(queueSize, fileWriterThreadCount, processorThreadCount);

	volatile bool terminateAll = false;

	unsigned width, height, numberOfChannel, bitsPerChannel;
	CHECK_FAST(PreloadImage(
		options.InputPath,
		options.IsFolder,
		width, height, numberOfChannel, bitsPerChannel
	));
	
	options.MaxHeight = options.MaxHeight == 0 ? height : options.MaxHeight;
	options.MaxWidth = options.MaxWidth == 0 ? width : options.MaxWidth;
	options.SurfaceFmt = IdentifySurface(bitsPerChannel, numberOfChannel);

	printf("Surface format: %s\n", EnumToString(options.SurfaceFmt));
	printf("Sampling format: %s\n", EnumToString(options.JpegEncoder.SamplingFmt));
	printf("JPEG quality: %d%%\n", options.JpegEncoder.Quality);
	
	int restartInterval = 32;
	if (options.SurfaceFmt == FAST_RGB8 || options.SurfaceFmt == FAST_RGB12) {
		if (options.JpegEncoder.SamplingFmt == FAST_JPEG_444)
			restartInterval = 10;
		if (options.JpegEncoder.SamplingFmt == FAST_JPEG_422)
			restartInterval = 8;
		if (options.JpegEncoder.SamplingFmt == FAST_JPEG_420)
			restartInterval = 5;
	}
	if (options.SurfaceFmt == FAST_I8 || options.SurfaceFmt == FAST_I12)
	{
		options.JpegEncoder.SamplingFmt = FAST_JPEG_Y;
	}
	
	printf("Restart interval: %d\n", restartInterval);

	fastJpegQuantState_t quantState = { 0 };
	bool quantStateEnabled = false;
	if (options.JpegEncoder.QuantTableFileName != NULL) {
		printf("External quant table: %s\n", options.JpegEncoder.QuantTableFileName);
		CHECK_FAST(fvLoadQuantTable(options.JpegEncoder.QuantTableFileName, quantState));
		quantStateEnabled = true;
	}

	ManagedConstFastAllocator<0> imgMgr;
	imgMgr.initManager(
		batchSize * queueSize,
		(size_t)GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth) * (size_t)options.MaxHeight
	);
	
	ManagedConstFastAllocator<1> jfifMgr;
	jfifMgr.initManager(
		batchSize * queueSize,
		(size_t)GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth) * (size_t)options.MaxHeight
	);

	CollectionFastAllocator<0> singleImgMgr;
	singleImgMgr.initManager(
		batchSize);

	auto* jpegWriter = new JpegAsyncFileWriter<ManagedConstFastAllocator<1>>();
	jpegWriter->Init(batchSize, queueSize, fileWriterThreadCount, processorThreadCount, options.Discard, &terminateAll);
	
	AsyncReaderTaskEnqueuer* fileEnqueuer = nullptr;
	BaseOptions* boptions = &options;

	IAsyncFileReader* reader = nullptr;
	IAsyncProcessor* processor = nullptr;

	AsyncProcessorFactory<
		EncoderAsync<PortableAsyncFileReader<ManagedConstFastAllocator<0>>>,
		EncoderAsync<PortableAsyncSingleFileReader<CollectionFastAllocator<0>>>,
		PortableAsyncFileReader<ManagedConstFastAllocator<0>>,
		PortableAsyncSingleFileReader<CollectionFastAllocator<0>>,
		JpegAsyncFileWriter<ManagedConstFastAllocator<1>> >
		(&fileEnqueuer, &reader, &processor,
			jpegWriter, boptions, batchSize,
			queueSize, fileReaderThreadCount,
			processorThreadCount, &terminateAll);

/*	
	auto* ppmReader = new PortableAsyncFileReader<ManagedConstFastAllocator<0>>();
		ppmReader->Init(batchSize, queueSize, fileReaderThreadCount, processorThreadCount, &terminateAll);

	AsyncReaderTaskEnqueuer* fileEnqueuer = AsyncReaderTaskEnqueuerFactory(&options, ppmReader, &terminateAll);

	auto processor = new AsyncProcessor<EncoderAsync, PortableAsyncFileReader<ManagedConstFastAllocator<0>>, JpegAsyncFileWriter<ManagedConstFastAllocator<1>>>();

	fastStatus_t status = processor->Init(
		processorThreadCount, &options,
		ppmReader, jpegWriter,
		&terminateAll,
		quantStateEnabled ? &quantState : nullptr
	);
	if (status != FAST_OK)
	{
		printf("!!!ERROR!!!! %d\n", status);
		terminateAll = true;
	}*/

	processor->WaitAll();
	reader->WaitAll();
	jpegWriter->WaitAll();
	if (fileEnqueuer != nullptr)
		fileEnqueuer->WaitAll();

	if (!terminateAll) {
		PrintProcessorResults(processor, &options, "encode");
	}
	
	delete reader;
	delete jpegWriter;
	delete processor;
	if (fileEnqueuer != nullptr)
		delete fileEnqueuer;

	jfifMgr.freeAll();
	imgMgr.freeAll();
	singleImgMgr.freeAll();

	return FAST_OK;
}
