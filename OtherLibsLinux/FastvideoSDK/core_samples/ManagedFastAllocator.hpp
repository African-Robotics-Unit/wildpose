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
#include "BaseAllocator.h"
#include "fastvideo_sdk.h"
#include "FastAllocator.h"

#include <vector>
#include <list>

#include <cstdio>
#include <mutex>

#include <algorithm>

struct MemoryItem {
	void* p;
	size_t size;
};

template<typename Allocator>
class VariableMemoryManager {
	std::vector<MemoryItem> allBuffers;
	unsigned int count;
	size_t maxSize;
	std::list<MemoryItem> freeBuffers;

	bool isInited;
	std::mutex m;
	
	const float allocationMul = 1.2f;

public:
	VariableMemoryManager() {
		isInited = false;
		count = 0;
		maxSize = 0;
	}

	VariableMemoryManager(const VariableMemoryManager<Allocator>& copy) {
		isInited = copy.isInited;
		count = copy.count;
		maxSize = copy.maxSize;
		
		freeBuffers = copy.freeBuffers;
		allBuffers = copy.allBuffers;
	}

	void init(unsigned int count) {
		this->count = count;
		allBuffers.reserve(count);

		isInited = true;
	}

	bool IsInited() {
		return this->isInited;
	}

	void* allocate(size_t bytesCount) {
		std::lock_guard<std::mutex> lock(m);

		if (bytesCount > maxSize)
			maxSize = bytesCount;

		if (freeBuffers.size() == 0) {
			if (allBuffers.size() > count)
				throw std::bad_alloc();

			Allocator alloc;
			MemoryItem newItem;
			{
				newItem.size = maxSize;
				newItem.p = alloc.allocate(newItem.size);
			}
			allBuffers.push_back(newItem);

			std::sort(allBuffers.begin(), allBuffers.end(), [](const MemoryItem& item, const MemoryItem& item2) {
				return item.p < item2.p;
				}); // Linux alloc memory block in random order.

			return newItem.p;
		}
		
		auto freeBuffer = std::lower_bound(freeBuffers.begin(), freeBuffers.end(), bytesCount, [](const MemoryItem& item, size_t bytesCount) {
			return item.size < bytesCount;
		});

		if (freeBuffer != freeBuffers.end()) {
			void* p = freeBuffer->p;
			freeBuffers.erase(freeBuffer);
			return p;
		}

		// no sutable element found
		// remove smallest - allocate max
		void *removing = freeBuffers.begin()->p;
		freeBuffers.erase(freeBuffers.begin());

		auto removedItem = std::lower_bound(allBuffers.begin(), allBuffers.end(), removing, [](const MemoryItem& item, void* p) {
			return item.p < p; 
		});

		Allocator alloc;
		alloc.deallocate(removedItem->p);
		
		removedItem->size = maxSize;
		void* p = alloc.allocate(maxSize);
		removedItem->p = p;

		std::sort(allBuffers.begin(), allBuffers.end(), [](const MemoryItem& item, const MemoryItem& item2) {
			return item.p < item2.p;
		});

		return p;
	}

	void deallocate(void* p) {
		std::lock_guard<std::mutex> lock(m);
		//freeBuffers.push(p);

		auto deallocateItem = std::lower_bound(allBuffers.begin(), allBuffers.end(), p, [](const MemoryItem& item, void* p) {
			return item.p < p;
		});

		auto insertAfterItem = std::lower_bound(freeBuffers.begin(), freeBuffers.end(), deallocateItem->size, [](const MemoryItem& item, size_t size) {
			return item.size < size;
		});

		freeBuffers.insert(insertAfterItem, *deallocateItem);
	}

	void operator()(void* p) {
		deallocate(p);
	}

	void freeAll() {
		if (freeBuffers.size() != allBuffers.size())
			throw std::bad_alloc();

		Allocator alloc;
		for (int i = 0; i < allBuffers.size(); i++) {
			alloc.deallocate(allBuffers[i].p);
		}
		freeBuffers.clear();
		allBuffers.clear();

		isInited = false;
	}
};

extern std::vector<VariableMemoryManager<FastAllocator>> fastAllocatorVariableManagers;

template<int id>
class ManagedFastAllocator : public BaseAllocator {

public:
	void initManager(unsigned int count) {
		/*const int mgrIdx = id;
		printf("%d %d\n", mgrIdx, (int)fastAllocatorConstManagers.size() - 1 < mgrIdx);
		bool isResize = (int)fastAllocatorConstManagers.size() - 1 < mgrIdx;*/

		if ((int)fastAllocatorVariableManagers.size() - 1 < id) {
			fastAllocatorVariableManagers.resize(id + 1);
		}

		if (!fastAllocatorVariableManagers[id].IsInited()) {
			fastAllocatorVariableManagers[id].init(count);
		}
	}

	void* allocate(size_t bytesCount) {
		if (!fastAllocatorVariableManagers[id].IsInited())
			throw std::bad_alloc();

		return fastAllocatorVariableManagers[id].allocate(bytesCount);
	}

	void deallocate(void* p) {
		fastAllocatorVariableManagers[id].deallocate(p);
	}

	void operator()(void* p) {
		fastAllocatorVariableManagers[id].deallocate(p);
	}

	void freeAll() {
		fastAllocatorVariableManagers[id].freeAll();
	}
};
