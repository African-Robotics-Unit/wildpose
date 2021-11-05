/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#ifndef __SDI_COMMON__
#define __SDI_COMMON__

#include "fastvideo_sdk.h"

#include <memory>
#include "FastAllocator.h"

inline  fastStatus_t fvLoadBinary(const char *fileName, std::unique_ptr<unsigned char, FastAllocator> &buffer) {
	FILE *fp = fopen(fileName, "rb");
	if (!fp) {
		fprintf(stderr, "Cannot open source file\n");
		return FAST_IO_ERROR;
	}

	fseek(fp, 0L, SEEK_END);
	const long fileSize = ftell(fp);
	rewind(fp);

	FastAllocator alloc;
	buffer.reset(static_cast<unsigned char *>(alloc.allocate(fileSize)));

	unsigned char *ptr = buffer.get();
	const size_t readSize = fread(ptr, sizeof(unsigned char), fileSize, fp);
	fclose(fp);
	if (readSize != fileSize) {
		fprintf(stderr, "Cannot read full file\n");
		return FAST_IO_ERROR;
	}

	return FAST_OK;
}

inline fastStatus_t fvSaveBinary(const char *fileName, unsigned char *buffer, const unsigned bufferSize) {
	FILE *fp = fopen(fileName, "w+b");
	if (!fp) {
		fprintf(stderr, "Cannot create destination file\n");
		return FAST_IO_ERROR;
	}

	const size_t writeSize = fwrite(buffer, sizeof(unsigned char), bufferSize, fp);

	fclose(fp);
	if (writeSize != bufferSize) {
		fprintf(stderr, "Cannot store full file\n");
		return FAST_IO_ERROR;
	}

	return FAST_OK;
}

bool LoadBinary(const char *fileName, std::unique_ptr<unsigned char, FastAllocator> &buffer);

#endif // __SDI_COMMON__