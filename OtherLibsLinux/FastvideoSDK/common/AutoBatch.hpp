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

#include "Batch.hpp"
#include "BatchedQueue.h"

#include <mutex>

class AtomicCounter {
private:
	
	volatile int cnt;
	std::mutex m;
public:
	AtomicCounter(void) : cnt(1) {
	}

	void Add(void) {
		m.lock();
		cnt++;
		m.unlock();
	}

	int Release(void) {
		m.lock();
		--cnt;
		m.unlock();
		return cnt;
	}
};



template<typename T>
class AutoBatch {
protected:
	Batch<T>* currentBatch;
	BatchedQueue<T>* currentQueue;
	unsigned int threadId;
	AtomicCounter* cnt;

	virtual void Release() {

	}

	void Copy(const AutoBatch& src)
	{
		currentQueue = src.currentQueue;
		currentBatch = src.currentBatch;
		threadId = src.threadId;
		cnt = src.cnt;
	}

	void Destroy() {
		if (cnt == NULL) return;

		if (cnt->Release() == 0) {
			Release();
			delete cnt;
			cnt = NULL;
		}
	}

public:

	AutoBatch(BatchedQueue<T>* currentQueue, Batch<T>* currentBatch, unsigned int threadId, AtomicCounter* cnt)
	{
		this->currentQueue = currentQueue;
		this->currentBatch = currentBatch;
		this->threadId = threadId;
		this->cnt = cnt;
	}

	AutoBatch(void) :
		currentBatch(nullptr),
		currentQueue(nullptr),
		threadId(0),
		cnt(NULL) {
	}

	 AutoBatch(const AutoBatch<T>& src) {
		Copy(src);

		if (cnt != NULL)
			cnt->Add();
	}

	T* At(int i) {
		if (currentBatch == nullptr)
			throw;
		return currentBatch->At(i);
	}

	unsigned int GetSize() const
	{
		if (currentBatch == nullptr)
			throw;
		return currentBatch->GetSize();
	}

	unsigned int GetFilledItem() const
	{
		if (currentBatch == nullptr)
			throw;
		return currentBatch->GetFilledItem();
	}

	void SetFilltedItem(unsigned int count)
	{
		if (currentBatch == nullptr)
			throw;
		currentBatch->SetFilltedItem(count);
	}

	bool IsEmpty()
	{
		return currentBatch == nullptr;
	}
};


template<typename T>
class ReaderAutoBatch : public  AutoBatch <T> {
protected:
	using AutoBatch <T>::currentQueue;
	using AutoBatch <T>::threadId;
	using AutoBatch <T>::Destroy;
	void Release() {
		if (currentQueue != nullptr)
			currentQueue->FreeReaderBatch(threadId);
	}
public:
	ReaderAutoBatch(BatchedQueue<T>* currentQueue, Batch<T>* currentBatch, unsigned int threadId, AtomicCounter* cnt) :
		AutoBatch<T>(currentQueue, currentBatch, threadId, cnt) {
	}

	~ReaderAutoBatch()	{
		Destroy();
	}
};


template<typename T>
class WriterAutoBatch : public  AutoBatch <T> {
protected:
	using AutoBatch <T>::currentQueue;
	using AutoBatch <T>::threadId;
	using AutoBatch <T>::Destroy;

	void Release() {
		if (currentQueue != nullptr)
			currentQueue->CommitWriterBatch(threadId);
	}
public:
	WriterAutoBatch(BatchedQueue<T>* currentQueue, Batch<T>* currentBatch, unsigned int threadId, AtomicCounter* cnt) :
		AutoBatch<T>(currentQueue, currentBatch, threadId, cnt) {
	}

	~WriterAutoBatch() {
		Destroy();
	}
};