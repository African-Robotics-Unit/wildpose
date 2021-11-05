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

#include "J2kDecoderAsync.h"

#include "checks.h"

#include "helper_bytestream.hpp"

#include "JpegDecoderSampleOptions.h"

#include "supported_files.hpp"

#include "BatchedQueue.h"
#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"
#include "CollectionFastAllocator.hpp"

#include "AsyncReaderTaskEnqueuer.hpp"
#include "AsyncFileReader.hpp"
#include "AsyncFileWriter.hpp"
#include "AsyncProcessor.hpp"

#include "PreloadJ2k.hpp"
#include "MultiThreadInfoPrinter.h"
#include "AsyncReaderTaskEnqueuerFactory.hpp"
#include "AsyncProcessorFactory.hpp"

fastStatus_t RunJ2kDecoderAsync(J2kDecoderOptions options) {

	fastSdkParametersHandle_t hSdkParameters;
	{
		CHECK_FAST(fastGetSdkParametersHandle(&hSdkParameters));
		CHECK_FAST(fastDecoderJ2kLibraryInit(hSdkParameters));
	}

	const int fileReaderThreadCount = options.NumberOfReaderThreads;
	const int processorThreadCount = options.NumberOfThreads;
	const int fileWriterThreadCount = options.NumberOfWriterThreads;

	const int batchSize =  options.BatchSize;

	unsigned queueSize = CheckAndUpdateQueueSize(options.Queue, processorThreadCount, fileReaderThreadCount);
	queueSize = CheckAndUpdateQueueSize(queueSize, fileWriterThreadCount, processorThreadCount);

	volatile bool terminateAll = false;

	fastJ2kImageInfo_t j2kInfo = { };
	CHECK_FAST(PreloadJ2k(options.InputPath, options.IsFolder, &j2kInfo));
	
	options.MaxHeight = options.MaxHeight == 0 ? j2kInfo.height : options.MaxHeight;
	options.MaxWidth = options.MaxWidth == 0 ? j2kInfo.width : options.MaxWidth;

	options.SurfaceFmt = j2kInfo.decoderSurfaceFmt;

	ManagedFastAllocator<0> j2kMgr;
	j2kMgr.initManager(batchSize * queueSize);

	CollectionFastAllocator<0> singleImgMgr;
	singleImgMgr.initManager(
		batchSize);

	ManagedConstFastAllocator<0> imgMgr;
	imgMgr.initManager(
		batchSize * queueSize,
		(size_t)GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth) * (size_t)options.MaxHeight
	);

	PortableAsyncFileWriter<ManagedConstFastAllocator<0>>* imgWriter = new PortableAsyncFileWriter<ManagedConstFastAllocator<0>>();
	imgWriter->Init(batchSize, queueSize, fileWriterThreadCount, processorThreadCount, options.Discard, &terminateAll);
	
	AsyncReaderTaskEnqueuer* fileEnqueuer = nullptr;
	BaseOptions* boptions = &options;

	IAsyncFileReader* reader = nullptr;
	IAsyncProcessor* processor = nullptr;

	AsyncProcessorFactory<
		J2kDecoderAsync<BytestreamAsyncFileReader<ManagedFastAllocator<0>>>,
		J2kDecoderAsync<BytestreamAsyncSingleFileReader<CollectionFastAllocator<0>>>,
		BytestreamAsyncFileReader<ManagedFastAllocator<0>>,
		BytestreamAsyncSingleFileReader<CollectionFastAllocator<0>>,
		PortableAsyncFileWriter<ManagedConstFastAllocator<0>>>
		(&fileEnqueuer, &reader, &processor,
			imgWriter, boptions, batchSize,
			queueSize, fileReaderThreadCount,
			processorThreadCount, &terminateAll);

	
	if (fileEnqueuer != nullptr)
		fileEnqueuer->WaitAll();

	processor->WaitAll();
	reader->WaitAll();
	imgWriter->WaitAll();

	if (!terminateAll) {
		PrintProcessorResults(processor, &options, "J2K Decode");
	}
	
	delete reader;
	delete imgWriter;
	delete processor;

	if (fileEnqueuer != nullptr)
		delete fileEnqueuer;

	j2kMgr.freeAll();
	imgMgr.freeAll();
	singleImgMgr.freeAll();

	return FAST_OK;
}
