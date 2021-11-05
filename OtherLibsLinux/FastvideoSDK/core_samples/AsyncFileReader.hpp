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

#include "Image.h"
#include "BatchedQueue.h"
#include "AutoBatch.hpp"

#include <string>
#include <thread>

#include "IAsyncFileReader.hpp"

template<typename Allocator, template<typename > class BatchElement>
class AsyncFileReader: public IAsyncFileReader {
protected:
	BatchedQueue<File_t>* fileTasks;
	BatchedQueue<BatchElement<Allocator>>* loadedFiles;

	std::thread** fileReaderThreads;

	unsigned int fileReaderThreadCount;
	unsigned int processorThreadCount;
	volatile bool* terminateAll;

	bool isHeaderToBytestream;

	Batch<File_t>* currentFileNameBatch;
	unsigned int currentFileNameBatchCount;

	void FileReaderThread(int threadId)
	{
		while (!(*terminateAll)) {
			Batch<File_t>* fileTaskBatch = fileTasks->GetNextReaderBatch(threadId);
			if (fileTaskBatch == nullptr)
				break;

			Batch<BatchElement<Allocator>>* loadedFileBatch = loadedFiles->GetNextWriterBatch(threadId);
			loadedFileBatch->SetFilltedItem(fileTaskBatch->GetFilledItem());

			for (int j = 0; j < fileTaskBatch->GetFilledItem() && !(*terminateAll); j++) {
				loadedFileBatch->At(j)->inputFileName = fileTaskBatch->At(j)->inputFileName;
				loadedFileBatch->At(j)->outputFileName = fileTaskBatch->At(j)->outputFileName;

				ReadFile(loadedFileBatch->At(j)->inputFileName, loadedFileBatch->At(j), isHeaderToBytestream);
			}
			fileTasks->FreeReaderBatch(threadId);
			loadedFiles->CommitWriterBatch(threadId);
		}
		loadedFiles->WriterFinished(threadId);
	}

	virtual void ReadFile(
		std::string fname,
		BatchElement<Allocator> *element,
		bool isHeaderToBytestream
	)
	{	}

public:
	AsyncFileReader()
	{

	}
	
	void Init(
		unsigned int batchSize, unsigned int batchCount,
		unsigned int fileReaderThreadCount, unsigned int processorThreadCount,
		volatile bool* terminateAll,
		bool isHeaderToBytestream = false
	) {
		this->fileReaderThreadCount = fileReaderThreadCount;
		this->processorThreadCount = processorThreadCount;
		this->terminateAll = terminateAll;
		this->isHeaderToBytestream = isHeaderToBytestream;

		fileTasks = new BatchedQueue<File_t>(
			batchCount, batchSize,
			terminateAll,
			1,
			fileReaderThreadCount);

		loadedFiles = new BatchedQueue<BatchElement<Allocator>>(
			batchCount, batchSize,
			terminateAll,
			fileReaderThreadCount,
			processorThreadCount);

		fileReaderThreads = new std::thread * [fileReaderThreadCount];
		for (int i = 0; i < fileReaderThreadCount; i++) {
			fileReaderThreads[i] = new std::thread(&AsyncFileReader::FileReaderThread, this, i);
		}

		currentFileNameBatchCount = 0;
	}

	void EnqueNextFile(File_t *fileInfo)
	{
		if (fileInfo == nullptr)
		{
			if (currentFileNameBatch != nullptr)
			{
				currentFileNameBatch->SetFilltedItem(currentFileNameBatchCount);
				fileTasks->CommitWriterBatch(0);
			}
			fileTasks->WriterFinished(0);
			return;
		}

		if (currentFileNameBatch == nullptr)
		{
			currentFileNameBatch = fileTasks->GetNextWriterBatch(0);
			currentFileNameBatchCount = 0;
		}

		File_t *fileTask = currentFileNameBatch->At(currentFileNameBatchCount);
		fileTask->inputFileName = fileInfo->inputFileName;
		fileTask->outputFileName = fileInfo->outputFileName;
		currentFileNameBatchCount++;

		if (currentFileNameBatch->GetSize() == currentFileNameBatchCount)
		{
			fileTasks->CommitWriterBatch(0);
			currentFileNameBatch = nullptr;
		}
		return;
	}

	ReaderAutoBatch<BatchElement<Allocator>> ReadNextFileBatch(unsigned int threadId)
	{
		AtomicCounter* cnt = new AtomicCounter();
		Batch<BatchElement<Allocator>> *loadedFile = loadedFiles->GetNextReaderBatch(threadId);
		ReaderAutoBatch<BatchElement<Allocator>>  readerBatch(loadedFiles, loadedFile, threadId, cnt);
		return readerBatch;
	}

	void WaitAll() {
		for (int i = 0; i < fileReaderThreadCount; i++) {
			fileReaderThreads[i]->join();
		}
	}

	~AsyncFileReader()
	{
		for (int i = 0; i < fileReaderThreadCount; i++) {
			delete fileReaderThreads[i];
		}
		delete[] fileReaderThreads;

		delete fileTasks;
		delete loadedFiles;
	}
};

template<typename Allocator >
class PortableAsyncFileReader : public AsyncFileReader<Allocator, Image> {
	void ReadFile(std::string fname, Image<Allocator>* element, bool isHeaderToBytestream) {
		fvLoadImage(
			element->inputFileName,
			element->outputFileName,
			*element,
			element->h, element->w,
			element->bitsPerChannel,
			false
		);
	}
};

template<typename Allocator >
class JpegAsyncFileReader : public AsyncFileReader<Allocator, JfifInfo> {
	void ReadFile(std::string fname, JfifInfo<Allocator> *element, bool isHeaderToBytestream) {
		fastJfifLoadFromFile(fname, element, isHeaderToBytestream, nullptr);
		fastJfifFreeExif(&element->info);
	}
};

template<typename Allocator >
class BytestreamAsyncFileReader : public AsyncFileReader<Allocator, Bytestream> {
	void ReadFile(std::string fname, Bytestream<Allocator> *element, bool isHeaderToBytestream) {
		fvLoadBytestream(fname, *element, false);
	}
};
