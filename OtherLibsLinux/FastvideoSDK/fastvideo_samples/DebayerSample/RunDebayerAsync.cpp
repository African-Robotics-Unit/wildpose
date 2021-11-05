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

#include "DebayerAsync.h"

#include "checks.h"

#include "EnumToStringSdk.h"

#include "helper_bytestream.hpp"
#include "helper_pfm.hpp"
#include "supported_files.hpp"

#include "BatchedQueue.h"
#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"

#include "AsyncReaderTaskEnqueuer.hpp"
#include "AsyncFileReader.hpp"
#include "AsyncFileWriter.hpp"
#include "AsyncProcessor.hpp"

#include "PreloadUncompressedImage.hpp"
#include "MultiThreadInfoPrinter.h"
#include "AsyncReaderTaskEnqueuerFactory.hpp"

fastSurfaceFormat_t GetOutputSurfaceFormat(const fastSurfaceFormat_t surfaceFmt) {
	switch (surfaceFmt) {
		case FAST_I12:
			return FAST_RGB12;
		case FAST_I16:
			return FAST_RGB16;
		default:
			return FAST_RGB8;
	}
}

fastStatus_t RunDebayerAsync(DebayerSampleOptions &options) {
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

	printf("Input surface format: grayscale\n");
	printf("Pattern: %s\n", EnumToString(options.Debayer.BayerFormat));
	printf("Output surface format: %s\n", EnumToString(IdentifySurface(GetBitsPerChannelFromSurface(options.SurfaceFmt), 3)));
	printf("Debayer algorithm: %s\n", EnumToString(options.Debayer.BayerType));

	ImageT<float, FastAllocator> matrixA;
	if (options.GrayscaleCorrection.MatrixA != nullptr) {
		unsigned channels;
		bool failed = false;

		printf("\nMatrix A: %s\n", options.GrayscaleCorrection.MatrixA);
		CHECK_FAST(fvLoadPFM(options.GrayscaleCorrection.MatrixA, matrixA.data, matrixA.w, matrixA.wPitch, FAST_ALIGNMENT, matrixA.h, channels));

		if (channels != 1) {
			fprintf(stderr, "Matrix A file must not be color\n");
			failed = true;
		}

		if (options.MaxHeight != matrixA.h || options.MaxWidth != matrixA.w) {
			fprintf(stderr, "Input and matrix A file parameters mismatch\n");
			failed = true;
		}

		if (failed) {
			fprintf(stderr, "Matrix A file reading error. Ignore parameters\n");
			failed = false;
		}
	}

	ImageT<char, FastAllocator> matrixB;
	if (options.GrayscaleCorrection.MatrixB != nullptr) {
		bool failed = false;

		printf("\nMatrix B: %s\n", options.GrayscaleCorrection.MatrixB);
		CHECK_FAST(fvLoadImage(std::string(options.GrayscaleCorrection.MatrixB), std::string(""), matrixB, options.MaxHeight, options.MaxWidth, 8, false));

		if (matrixB.surfaceFmt != FAST_I8 && matrixB.surfaceFmt != FAST_I12 && matrixB.surfaceFmt != FAST_I16) {
			fprintf(stderr, "Matrix B file must not be color\n");
			failed = true;
		}

		if (options.MaxHeight != matrixB.h || options.MaxWidth != matrixB.w) {
			fprintf(stderr, "Input and matrix B file parameters mismatch\n");
			failed = true;
		}

		if (failed) {
			fprintf(stderr, "Matrix B file reading error. Ignore parameters\n");
		}
	}
	
	fastSam_t madParameter = { 0 };
	{
		madParameter.correctionMatrix = options.GrayscaleCorrection.MatrixA != nullptr ? matrixA.data.get() : nullptr;
		madParameter.blackShiftMatrix = options.GrayscaleCorrection.MatrixB != nullptr ? matrixB.data.get() : nullptr;
	}
	bool madEnabled =
		options.GrayscaleCorrection.MatrixA != nullptr ||
		options.GrayscaleCorrection.MatrixB != nullptr;

	ManagedConstFastAllocator<0> imgReaderMgr;
	imgReaderMgr.initManager(
		batchSize * queueSize,
		(size_t)GetPitchFromSurface(GetOutputSurfaceFormat(options.SurfaceFmt), options.MaxWidth) * (size_t)options.MaxHeight
	);

	ManagedConstFastAllocator<1> imgWriterMgr;
	imgWriterMgr.initManager(
		batchSize * queueSize,
		(size_t)GetPitchFromSurface(GetOutputSurfaceFormat(options.SurfaceFmt), options.MaxWidth) * (size_t)options.MaxHeight
	);
	
	auto* imgReader = new PortableAsyncFileReader<ManagedConstFastAllocator<0>>();
	imgReader->Init(batchSize, queueSize, fileReaderThreadCount, processorThreadCount, &terminateAll);

	auto* imgWriter = new PortableAsyncFileWriter<ManagedConstFastAllocator<1>>();
	imgWriter->Init(batchSize, queueSize, fileWriterThreadCount, processorThreadCount, options.Discard, &terminateAll);

	AsyncReaderTaskEnqueuer* fileEnqueuer = AsyncReaderTaskEnqueuerFactory(&options, imgReader, &terminateAll);

	auto processor = new AsyncProcessor<DebayerAsync, PortableAsyncFileReader<ManagedConstFastAllocator<0>>, PortableAsyncFileWriter<ManagedConstFastAllocator<1>>>();
	fastStatus_t status = processor->Init(
		processorThreadCount, &options,
		imgReader, imgWriter,
		&terminateAll,
		madEnabled ? &madParameter : nullptr
	);
	if (status != FAST_OK)
		terminateAll = true;
	
	fileEnqueuer->WaitAll();
	processor->WaitAll();
	imgReader->WaitAll();
	imgWriter->WaitAll();

	if (!terminateAll) {
		PrintProcessorResults(processor, &options, "debayer");
	}
	
	delete imgReader;
	delete imgWriter;
	delete fileEnqueuer;
	delete processor;

	imgReaderMgr.freeAll();
	imgWriterMgr.freeAll();

	return FAST_OK;
}
