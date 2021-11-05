/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <iostream>
#include <memory>

#include "fastvideo_sdk.h"
#include "FastAllocator.h"
#include "helper_common.h"
#include "cube.h"

template<class T, class Allocator> fastStatus_t
fvLoadLut(const char *file, std::unique_ptr<T, Allocator> &data, unsigned size) {
	FILE *fp = NULL;

	if (FOPEN_FAIL(FOPEN(fp, file, "r")))
		return FAST_IO_ERROR;

	Allocator alloc;
	data.reset((T*)alloc.allocate(size * sizeof(T)));

	unsigned i = 0;
	while (i < size && !feof(fp)) {
		float value;
		fscanf(fp, "%f", &value);
		data.get()[i] = static_cast<T>(value);
		i++;
	}

	if (i != size) {
		return FAST_IO_ERROR;
	}

	fclose(fp);
	return FAST_OK;
}

template fastStatus_t fvLoadLut(const char *file, std::unique_ptr<unsigned char, FastAllocator> &data, unsigned size);
template fastStatus_t fvLoadLut(const char *file, std::unique_ptr<unsigned short, FastAllocator> &data, unsigned size);
template fastStatus_t fvLoadLut(const char *file, std::unique_ptr<float, FastAllocator> &data, unsigned size);
