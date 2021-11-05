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

#include <string>

#include "Image.h"
#include "BatchedQueue.h"
#include "AutoBatch.hpp"
#include "helper_bytestream.hpp"
#include <thread>

#include "helper_bytestream.hpp"

template<typename Allocator, template<typename> class BatchElement>
class AsyncFileWriter {
protected:
	BatchedQueue<BatchElement<Allocator>>* imgs;

	std::thread** fileWriterThreads;

	unsigned int fileWriterThreadCount;
	unsigned int processorThreadCount;
	volatile bool* terminateAll;

	bool discard;

	Batch<BatchElement<Allocator>>* currentWriteBatch;
	unsigned int currentWriteBatchCount;

	void FileWriterFunc(int threadId) {
		while (!(*terminateAll)) {
			auto imgBatch = imgs->GetNextReaderBatch(threadId);
			if (imgBatch == nullptr)
				break;

			for (int j = 0; j < imgBatch->GetFilledItem() && !(*terminateAll); j++) {
				auto* img = imgBatch->At(j);
				if (!discard) {
					WriteFile(img);
				}
				img->ReleaseBuffer();
			}
			imgs->FreeReaderBatch(threadId);
		}
	}

	virtual void WriteFile(BatchElement<Allocator>* img) {
	}

public:
	AsyncFileWriter()
	{
		discard = false;
		terminateAll = nullptr;
		fileWriterThreads = nullptr;
	}

	void Init(unsigned int batchSize, unsigned int batchCount, unsigned int fileWriterThreadCount, unsigned int processorThreadCount, bool discard, volatile bool* terminateAll)
	{
		this->fileWriterThreadCount = fileWriterThreadCount;
		this->processorThreadCount = processorThreadCount;
		this->terminateAll = terminateAll;
		this->discard = discard;
		imgs = new BatchedQueue<BatchElement<Allocator>>(
			batchCount, batchSize,
			terminateAll,
			processorThreadCount,
			fileWriterThreadCount
		);

		fileWriterThreads = new std::thread * [fileWriterThreadCount];
		for (int i = 0; i < fileWriterThreadCount; i++) {
			fileWriterThreads[i] = new std::thread(&AsyncFileWriter::FileWriterFunc, this, i);
		}
	}

	WriterAutoBatch<BatchElement<Allocator>> GetNextWriterBatch(unsigned int threadId)
	{
		AtomicCounter* cnt = new AtomicCounter();
		Batch<BatchElement<Allocator>>* writerBatch = imgs->GetNextWriterBatch(threadId);
		WriterAutoBatch<BatchElement<Allocator>>  returnBatch(imgs, writerBatch, threadId, cnt);
		return returnBatch;
	}

	void WriterFinished(unsigned int threadId)
	{
		imgs->WriterFinished(threadId);
	}

	void WaitAll() {
		for (int i = 0; i < fileWriterThreadCount; i++) {
			fileWriterThreads[i]->join();
		}
	}

	~AsyncFileWriter()
	{
		for (int i = 0; i < fileWriterThreadCount; i++) {
			delete fileWriterThreads[i];
		}
		delete[] fileWriterThreads;

		delete imgs;
	}
};

template<typename Allocator >
class PortableAsyncFileWriter : public AsyncFileWriter<Allocator, Image> {
	void WriteFile(Image<Allocator>* img) {
		fvSaveImageToFile((char*)img->outputFileName.c_str(), img->data, img->surfaceFmt, img->bitsPerChannel, img->h, img->w, img->wPitch, false);
	}
};

template<typename Allocator >
class JpegAsyncFileWriter : public AsyncFileWriter<Allocator, JfifInfo> {
	void WriteFile(JfifInfo<Allocator>* img) {
		fastJfifStoreToFile(img->outputFileName.c_str(), &img->info);
	}
};

template<typename Allocator >
class BytestreamAsyncFileWriter : public AsyncFileWriter<Allocator, Bytestream> {
	void WriteFile(Bytestream<Allocator>* img) {
		fvSaveBytestream(
			img->outputFileName,
			img->data.get(),
			img->size,
			false
		);
	}
};
