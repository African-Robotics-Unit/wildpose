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
#include "fast_memory.h"
#include <cuda_runtime.h>

class FastAllocatorPinned {
public:
	typedef size_t		size_type;
	typedef const void*	const_pointer;

	template<class T>
	static T *allocate(size_t bytesCount, const_pointer = 0) {
		T* p = NULL;
		cudaError_t ret_cuda = cudaMallocHost( (void **)&p,  bytesCount * sizeof(T), cudaHostAllocMapped);
		if ( ret_cuda != cudaSuccess ) {
			throw std::bad_alloc();
		}
		return p;
	}

	static void deallocate(void* p, size_type) {
		cudaError_t ret_cuda = cudaFreeHost( p );
		if ( ret_cuda != cudaSuccess ) {
			throw std::bad_alloc();
		}
	}
	
	void operator()(void* p) {
		cudaError_t ret_cuda = cudaFreeHost( p );
		if ( ret_cuda != cudaSuccess ) {
			throw std::bad_alloc();
		}
	}
};
