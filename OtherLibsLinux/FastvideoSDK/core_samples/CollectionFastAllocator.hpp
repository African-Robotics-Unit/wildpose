#pragma once

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



template<typename Allocator>
class CollectionMemoryManager {
	std::list<void*> allocatedBuffer;
	std::mutex m;

public:
	CollectionMemoryManager() {
	
	}

	CollectionMemoryManager(const CollectionMemoryManager<Allocator>& copy) {
		allocatedBuffer = copy.allocatedBuffer;
	}

	void init() {
		
	}

	bool IsInited() {
		return true;
	}

	void* allocate(size_t bytesCount) {
		std::lock_guard<std::mutex> lock(m);

		Allocator alloc;
		void* p = alloc.allocate(bytesCount);
		allocatedBuffer.push_back(p);
		
		return p;
	}

	void deallocate(void* p) {
		
	}

	void operator()(void* p) {
		
	}

	void freeAll() {
		if (allocatedBuffer.size() == 0)
			return;
		Allocator alloc;
		for (auto i = allocatedBuffer.begin(); i != allocatedBuffer.end(); i++) {
			alloc.deallocate(*i);
			
		}
	}

	~CollectionMemoryManager() {
		//freeAll();
	}
};

extern std::vector<CollectionMemoryManager<FastAllocator>> fastAllocatorCollectionManagers;

template<int id >
class CollectionFastAllocator : public BaseAllocator {

public:
	void initManager(unsigned int count) {
		/*const int mgrIdx = id;
		printf("%d %d\n", mgrIdx, (int)fastAllocatorConstManagers.size() - 1 < mgrIdx);
		bool isResize = (int)fastAllocatorConstManagers.size() - 1 < mgrIdx;*/

		if ((int)fastAllocatorCollectionManagers.size() - 1 < id) {
			fastAllocatorCollectionManagers.resize(id + 1);
		}
	}

	void* allocate(size_t bytesCount) {
		if (!fastAllocatorCollectionManagers[id].IsInited())
			throw std::bad_alloc();

		return fastAllocatorCollectionManagers[id].allocate(bytesCount);
	}

	void deallocate(void* p) {
		fastAllocatorCollectionManagers[id].deallocate(p);
	}

	void operator()(void* p) {
		fastAllocatorCollectionManagers[id].deallocate(p);
	}

	void freeAll() {
		fastAllocatorCollectionManagers[id].freeAll();
	}
};
