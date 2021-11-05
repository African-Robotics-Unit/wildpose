/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "fastvideo_sdk.h"
#include <string>
#include <memory>
#include <cstring>
#include "alignment.hpp"

template<class T, class Allocator>
class Image {
public:
	std::string inputFileName;
	std::string outputFileName;

	std::unique_ptr<T, Allocator> data;
	unsigned w;
	unsigned h;
	unsigned wPitch;
	unsigned bitsPerChannel;

	fastSurfaceFormat_t surfaceFmt;
	fastJpegFormat_t samplingFmt;
	bool isRaw;

	Image(void) {
		w = h = wPitch = 0;
		bitsPerChannel = 8;
		isRaw = false;
	};

	Image(const Image &img) {
		inputFileName = img.inputFileName;
		outputFileName = img.outputFileName;
		w = img.w;
		h = img.h;
		wPitch = img.wPitch;
		bitsPerChannel = img.bitsPerChannel;
		surfaceFmt = img.surfaceFmt;
		samplingFmt = img.samplingFmt;

		size_t fullSize = (size_t)wPitch * h;
		isRaw = img.isRaw;

		try {
			Allocator alloc;
			data.reset((T*)alloc.allocate(fullSize));
		} catch (std::bad_alloc& ba) {
			fprintf(stderr, "Memory allocation failed: %s\n", ba.what());
			throw;
		}
		memcpy(data.get(), img.data.get(), fullSize * sizeof(T));
	};

	unsigned GetBytesPerPixel() const {
		return uDivUp(bitsPerChannel, 8u);
	}
};

template<class Allocator >
class Bytestream {
public:
	std::string inputFileName;
	std::string outputFileName;

	std::unique_ptr<unsigned char, Allocator> data;
	size_t size;

	bool encoded;
	float loadTimeMs;

	Bytestream(void) { };

	Bytestream(const Bytestream &img) {
		inputFileName = img.inputFileName;
		outputFileName = img.outputFileName;
		size = img.size;
		encoded = img.encoded;
		loadTimeMs = img.loadTimeMs;

		try {
			Allocator alloc;
			data.reset((unsigned char*)alloc.allocate(size));
		} catch (std::bad_alloc& ba) {
			fprintf(stderr, "Memory allocation failed: %s\n", ba.what());
			return;
		}
		memcpy(data.get(), img.data.get(), size * sizeof(unsigned char));
	};
};

typedef struct {
	unsigned char *data;

	unsigned width;
	unsigned height;
	unsigned pitch;

	unsigned bitsPerChannel;
	unsigned channels;
} Image_t;

#endif
