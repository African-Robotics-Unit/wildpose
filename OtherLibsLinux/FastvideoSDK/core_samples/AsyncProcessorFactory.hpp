/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#pragma once

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


template<
	typename ProcessorForDir, typename ProcessorForSingle, 
	typename DirReader, typename SingleReader, 
	typename Writer>
void AsyncProcessorFactory(
	AsyncReaderTaskEnqueuer** fileEnqueuer,
	IAsyncFileReader** reader, 
	IAsyncProcessor** processor,
	
	Writer *writer, 
	BaseOptions* options, 
	unsigned batchSize, 
	unsigned  queueSize,  
	unsigned fileReaderThreadCount,
	unsigned processorThreadCount, 
	volatile bool* terminateAll)
{

	if (options->IsFolder) {

		auto* imgReader = new DirReader();
		imgReader->Init(batchSize, queueSize, fileReaderThreadCount, processorThreadCount, terminateAll);

		AsyncReaderTaskEnqueuerForDirWithRepeat* dirFileEnqueuer = new AsyncReaderTaskEnqueuerForDirWithRepeat();
		dirFileEnqueuer->Init(options->InputPath, options->OutputPath, options->RepeatCount, imgReader, terminateAll);
		*fileEnqueuer = dirFileEnqueuer;
		auto dirprocessor = new AsyncProcessor<ProcessorForDir, DirReader, Writer>();

		fastStatus_t status = dirprocessor->Init(processorThreadCount, options, imgReader, writer, terminateAll);
		if (status != FAST_OK)
			*terminateAll = true;
		*reader = imgReader;
		*processor = dirprocessor;
	}
	else {

		auto* imgReader = new SingleReader();
		imgReader->Init(options->InputPath, options->OutputPath, options->RepeatCount, batchSize, processorThreadCount);

		auto singleprocessor = new AsyncProcessor<ProcessorForSingle, SingleReader,	Writer>();
		*fileEnqueuer = nullptr;

		fastStatus_t status = singleprocessor->Init(processorThreadCount, options, imgReader, writer, terminateAll);
		if (status != FAST_OK)
			*terminateAll = true;

		*reader = imgReader;
		*processor = singleprocessor;
	}
}
