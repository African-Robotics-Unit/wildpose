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

#include "DecoderAsync.h"

#include "checks.h"

#include "helper_bytestream.hpp"

#include "JpegDecoderSampleOptions.h"
#include "helper_jpeg.hpp"

#include "supported_files.hpp"

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

fastStatus_t RunJpegDecoderAsync(JpegDecoderSampleOptions options) {
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

	if (options.SurfaceFmt == FAST_I12 || options.SurfaceFmt == FAST_RGB12) {
		fprintf(stderr, "Unsupported surface format\n");
		return FAST_OK;
	}

	ManagedFastAllocator<0> jfifMgr;
	jfifMgr.initManager(batchSize * queueSize);

	ManagedConstFastAllocator<0> imgMgr;
	imgMgr.initManager(
		batchSize * queueSize,
		(size_t)GetPitchFromSurface(options.SurfaceFmt, options.MaxWidth) * (size_t)options.MaxHeight
	);

	CollectionFastAllocator<0> singleImgMgr;
	singleImgMgr.initManager(
		batchSize);

	PortableAsyncFileWriter<ManagedConstFastAllocator<0>>* ppmWriter = new PortableAsyncFileWriter<ManagedConstFastAllocator<0>>();
	ppmWriter->Init(batchSize, queueSize, fileWriterThreadCount, processorThreadCount, options.Discard, &terminateAll);

	AsyncReaderTaskEnqueuer* fileEnqueuer = nullptr;
	BaseOptions* boptions = &options;

	IAsyncFileReader* reader = nullptr;
	IAsyncProcessor* processor = nullptr;

	AsyncProcessorFactory<
		DecoderAsync<JpegAsyncFileReader<ManagedFastAllocator<0>>>,
		DecoderAsync<JpegAsyncSingleFileReader<CollectionFastAllocator<0>>>,
		JpegAsyncFileReader<ManagedFastAllocator<0>>,
		JpegAsyncSingleFileReader<CollectionFastAllocator<0>>,
		PortableAsyncFileWriter<ManagedConstFastAllocator<0>> >
		(&fileEnqueuer, &reader, &processor,
			ppmWriter, boptions, batchSize,
			queueSize, fileReaderThreadCount,
			processorThreadCount, &terminateAll);

	if (fileEnqueuer != nullptr)
		fileEnqueuer->WaitAll();

	processor->WaitAll();
	reader->WaitAll();
	ppmWriter->WaitAll();

	if (!terminateAll) {
		PrintProcessorResults(processor, &options, "decode");
	}
	
	delete reader;
	delete ppmWriter;
	delete processor;
	if (fileEnqueuer != nullptr)
		delete fileEnqueuer;

	jfifMgr.freeAll();
	imgMgr.freeAll();
	singleImgMgr.freeAll();

	return FAST_OK;
}
