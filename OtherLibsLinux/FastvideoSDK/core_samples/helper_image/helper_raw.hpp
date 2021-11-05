/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef HELPER_RAW_H
#define HELPER_RAW_H

#include <fstream>
#include <memory>

#include "fastvideo_sdk.h"
#include "helper_common.h"

template <class T, class Allocator>
fastStatus_t fvLoadRaw(
	const char *file, std::unique_ptr<T, Allocator> &data,
	unsigned int width, unsigned int height, unsigned int channels,
	unsigned bitsPerChannel
) {
	if (sizeof(T) != sizeof(unsigned char)) {
		return FAST_IO_ERROR;
	}

	FILE *fp = NULL;

	if (FOPEN_FAIL(FOPEN(fp, file, "rb")))
		return FAST_IO_ERROR;

	const size_t sizeInBits = width * height * channels * bitsPerChannel * sizeof(unsigned char);
	const size_t sizeInBytes = _uSnapUp<size_t>(sizeInBits, 8) / 8;

	Allocator alloc;
	data.reset((T *)alloc.allocate(sizeInBytes));

	if (fread(data.get(), sizeof(unsigned char), sizeInBytes, fp) < sizeInBytes) {
		return FAST_IO_ERROR;
	}

	fclose(fp);
	return FAST_OK;
}

template <class T>
fastStatus_t fvSaveRaw(
	const char *file, T *data,
	unsigned int width, unsigned int height, unsigned int channels,
	unsigned bitsPerChannel
) {
	if (sizeof(T) != sizeof(unsigned char)) {
		return FAST_IO_ERROR;
	}

	FILE *fp = NULL;
	if (FOPEN_FAIL(FOPEN(fp, file, "wb+")))
		return FAST_IO_ERROR;

	const size_t sizeInBits = width * channels * bitsPerChannel * sizeof(unsigned char);
	const size_t sizeInBytes = _uSnapUp<size_t>(sizeInBits, 8) / 8;

	for (unsigned i = 0; i < height; i++) {
		if (fwrite(&data[i * sizeInBytes], sizeof(unsigned char), sizeInBytes, fp) < sizeInBytes) {
			return FAST_IO_ERROR;
		}
	}

	fclose(fp);
	return FAST_OK;
}

#endif
