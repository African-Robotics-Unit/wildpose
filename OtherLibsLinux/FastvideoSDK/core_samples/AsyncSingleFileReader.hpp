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
#include <vector>
#include <thread>

#include "IAsyncFileReader.hpp"



template<typename Allocator, template<typename > class BatchElement>
class AsyncSingleFileReader : public IAsyncFileReader {
protected:
	Batch<BatchElement<Allocator>>** batches;
	std::vector <void*> *batchBuffer;
	unsigned int processorThreadCount;
	const char* file;
	const char* outPath;
	int repeat;
	std::mutex lockRead;

	virtual void ReadFile(
		std::string fname,
		BatchElement<Allocator>* element
	) = 0;


	virtual void* GetBuffer(
		BatchElement<Allocator>* element
	) = 0;
	

	virtual void SetBuffer(
		BatchElement<Allocator>* element,
		void* p
	) = 0;
	

public:
	AsyncSingleFileReader()
	{

	}

	void Init(
		const char* file, const char* outPath, unsigned repeat,
		unsigned int batchSize, unsigned int processorThreadCount
	) {
		this->file = file;
		this->outPath = outPath;
		this->repeat = repeat;
		this->processorThreadCount = processorThreadCount;

		batches = new Batch<BatchElement<Allocator>> * [processorThreadCount];
		batchBuffer = new std::vector <void*>(batchSize * processorThreadCount);

		for (int i = 0; i < processorThreadCount; i++)
		{
			batches[i] = new Batch<BatchElement<Allocator>>(batchSize);

			int idx = 0;
			for (int j = 0; j < batches[i]->GetSize(); j++) {
				batches[i]->At(j)->inputFileName = file;
				batches[i]->At(j)->outputFileName = generateOutputFileName(outPath, idx);
				idx++;
				ReadFile(batches[i]->At(j)->inputFileName, batches[i]->At(j));
				batchBuffer->at(batchSize * i + j) = GetBuffer(batches[i]->At(j));
			}

		}
	}

	ReaderAutoBatch<BatchElement<Allocator>> ReadNextFileBatch(unsigned int threadId)
	{
		std::unique_lock<std::mutex> locker(lockRead);
		const unsigned batchSize = batches[threadId]->GetSize();

		for (int j = 0; j < batchSize; j++) {
			SetBuffer(batches[threadId]->At(j), batchBuffer->at(threadId * batchSize + j));
		}

		AtomicCounter* cnt = new AtomicCounter();

		if (batches[threadId]->GetSize() > repeat)
			batches[threadId]->SetFilltedItem(repeat);
		
		ReaderAutoBatch<BatchElement<Allocator>>  readerBatch(nullptr, (repeat > 0) ? batches[threadId] : nullptr, threadId, cnt);
		
		repeat -= batches[threadId]->GetSize();

		return readerBatch;
	}

	void WaitAll() {

	}

	void EnqueNextFile(File_t* fileInfo)
	{

	}

	~AsyncSingleFileReader()
	{
		
	}
};

template<typename Allocator >
class PortableAsyncSingleFileReader : public AsyncSingleFileReader<Allocator, Image> {

	virtual void* GetBuffer(Image<Allocator>* element)
	{
		return element->data.get();
	}

	virtual void SetBuffer(Image<Allocator>* element, void* p)
	{
		element->data.reset((unsigned char*)p);
	}

	void ReadFile(std::string fname, Image<Allocator>* element) {
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
class JpegAsyncSingleFileReader : public AsyncSingleFileReader<Allocator, JfifInfo> {

	virtual void* GetBuffer(JfifInfo<Allocator>* element)
	{
		return element->bytestream.get();
	}

	virtual void SetBuffer(JfifInfo<Allocator>* element, void* p)
	{
		element->bytestream.reset((unsigned char*)p);
	}


	void ReadFile(std::string fname, JfifInfo<Allocator>* element) {
		fastJfifLoadFromFile(fname, element, false, nullptr);
		fastJfifFreeExif(&element->info);
	}
};

template<typename Allocator >
class BytestreamAsyncSingleFileReader : public AsyncSingleFileReader<Allocator, Bytestream> {

	virtual void* GetBuffer(Bytestream<Allocator>* element)
	{
		return element->data.get();
	}

	virtual void SetBuffer(Bytestream<Allocator>* element, void* p)
	{
		element->data.reset((unsigned char*)p);
	}

	void ReadFile(std::string fname, Bytestream<Allocator>* element) {
		fvLoadBytestream(fname, *element, false);
	}
};

