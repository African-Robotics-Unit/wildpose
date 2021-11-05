/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "SDICommon.hpp"
#include "stdio.h"
#include "FastAllocator.h"
#include <memory>


bool LoadBinary(const char *fileName, std::unique_ptr<unsigned char, FastAllocator> &buffer) {
	FILE *fp = fopen(fileName, "rb");
	if (!fp) {
		fprintf(stderr, "Cannot open source file\n");
		return false;
	}

	fseek(fp, 0L, SEEK_END);
	const long fileSize = ftell(fp);
	rewind(fp);

	FastAllocator alloc;
	buffer.reset((unsigned char *)alloc.allocate(fileSize));

	unsigned char *ptr = buffer.get();
	size_t readSize = fread(ptr, sizeof(unsigned char), fileSize, fp);
	fclose(fp);
	if (readSize != fileSize) {
		fprintf(stderr, "Cannot read full file\n");
		return false;
	}

	return true;
}