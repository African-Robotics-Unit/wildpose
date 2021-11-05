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

#include <condition_variable>

#include "Batch.hpp"

static unsigned CheckAndUpdateQueueSize(unsigned initialQueueSize, unsigned readerCount, unsigned writerCount)
{
	unsigned maxValue = std::max(readerCount, writerCount);
	for (unsigned i = 0; i < maxValue; i++)
	{
		unsigned val = initialQueueSize + i;
		if (val % readerCount == 0 && val % writerCount == 0)
			return val;
	}
	return initialQueueSize;
}

template<typename T>
class BatchedQueue {
private:
	Batch<T>** values;
	int size;
	int batchSize;

	int* writerIndexes;
	int* readerIndexes;
	int Writers;
	int Readers;

	volatile bool* Terminate;
	volatile bool* writerFinished;

	int GetNextWriterIndex(int index) const;
	int GetNextReaderIndex(int index) const;

public:
	BatchedQueue(int queueSize, int batchSize, volatile bool* terminate, int writers, int readers);
	~BatchedQueue();

	void Reinit();

	Batch<T>* At(int i);
	void CommitWriterBatch(int writerId);

	int GetSize() const;
	int GetBatchSize() const;

	bool CheckReaderBatchReady(int readerId, bool waitRequired) const;
	int GetWriterBatchFilledItemCount(int writerId) const;

	Batch<T>* GetNextWriterBatch(int writerId);
	Batch<T>* GetNextReaderBatch(int readerId, bool waitRequired = true);

	int GetMaxItemCount() const;
	int GetFilledItemCount();

	void WaitFirstQueueFill();
	void WaitAllFree(int readerId);

	void FreeReaderBatch(int readerId);

	void ReleaseAll();

	void WriterFinished(int writerId);
	bool AllWriterFinished();
};
