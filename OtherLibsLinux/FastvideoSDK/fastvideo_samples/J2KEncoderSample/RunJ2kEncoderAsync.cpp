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

#include "J2kEncoderAsync.h"

#include "checks.h"

#include "helper_bytestream.hpp"

#include "JpegEncoderSampleOptions.h"

#include "supported_files.hpp"

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
#include "AsyncReaderTaskEnqueuerFactory.hpp"
#include "AsyncSingleFileReader.hpp"

#include "AsyncProcessorFactory.hpp"

fastStatus_t RunJ2kEncoderAsync(J2kEncoderOptions options) {

	fastSdkParametersHandle_t hSdkParameters;
	{
		CHECK_FAST(fastGetSdkParametersHandle(&hSdkParameters));
		CHECK_FAST(fastEncoderJ2kLibraryInit(hSdkParameters));
	}

	const int fileReaderThreadCount = options.NumberOfReaderThreads;
	const int processorThreadCount = options.NumberOfThreads;
	const int fileWriterThreadCount = options.NumberOfWriterThreads;

	const int batchSize =  options.BatchSize;

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

	ManagedConstFastAllocator<0> imgMgr;
	imgMgr.initManager(
		batchSize * queueSize,
		(size_t)GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth) * (size_t)options.MaxHeight);

	CollectionFastAllocator<0> singleImgMgr;
	singleImgMgr.initManager(
		batchSize );

	ManagedConstFastAllocator<1> j2kMgr;
	j2kMgr.initManager(
		batchSize * queueSize,
		(size_t)GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth) * (size_t)options.MaxHeight
	);
	
	auto* j2kWriter = new BytestreamAsyncFileWriter<ManagedConstFastAllocator<1>>();
	j2kWriter->Init(batchSize, queueSize, fileWriterThreadCount, processorThreadCount, options.Discard, &terminateAll);

	AsyncReaderTaskEnqueuer* fileEnqueuer = nullptr;
	BaseOptions* boptions = &options;

	IAsyncFileReader *reader = nullptr;
	IAsyncProcessor* processor = nullptr;


	AsyncProcessorFactory<
		J2kEncoderAsync<PortableAsyncFileReader<ManagedConstFastAllocator<0>>>,
		J2kEncoderAsync<PortableAsyncSingleFileReader<CollectionFastAllocator<0>>>,
		PortableAsyncFileReader<ManagedConstFastAllocator<0>>,
		PortableAsyncSingleFileReader<CollectionFastAllocator<0>>,
		BytestreamAsyncFileWriter<ManagedConstFastAllocator<1>> >
		(&fileEnqueuer, &reader, &processor,
			j2kWriter, boptions, batchSize,
			queueSize, fileReaderThreadCount,
			processorThreadCount, &terminateAll);
		
		
	if (fileEnqueuer != nullptr)
		fileEnqueuer->WaitAll();

	processor->WaitAll();
	reader->WaitAll();
	j2kWriter->WaitAll();

	if (!terminateAll) {
		PrintProcessorResults(processor, &options, "J2K Decode");
	}
	
	delete reader;
	delete j2kWriter;
	delete processor;

	if (fileEnqueuer != nullptr)
		delete fileEnqueuer;
	

	imgMgr.freeAll();
	j2kMgr.freeAll();
	singleImgMgr.freeAll();

	return FAST_OK;
}
