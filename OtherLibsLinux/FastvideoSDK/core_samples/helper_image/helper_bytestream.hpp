/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __HELPER_BYTESTREAM_HPP__
#define __HELPER_BYTESTREAM_HPP__

#include <list>

#include "fastvideo_sdk.h"
#include "Image.h"

#include <fstream>
#include <sstream>

#include "helper_dir.hpp"
#include "timing.hpp"
#include "Image.h"
#include "checks.h"

template<class Allocator>
fastStatus_t fvLoadBytestream(std::string fname, Bytestream<Allocator>& inputImg, bool info) {
	std::ifstream input(fname.c_str(), std::ifstream::binary);
	if (input.is_open()) {
		input.exceptions(std::ios::eofbit | std::ios::failbit | std::ios::badbit);

		const hostTimer_t timer = hostTimerCreate();
		hostTimerStart(timer);

		input.seekg(0, std::ifstream::end);
		const std::streamoff length = input.tellg();
		input.seekg(0, std::ifstream::beg);

		inputImg.size = length;
		inputImg.inputFileName = fname;
		inputImg.encoded = false;

		Allocator alloc;
		inputImg.data.reset((unsigned char*)alloc.allocate(length));

		input.read((char*)inputImg.data.get(), length);
		input.close();

		const double loadTime = hostTimerEnd(timer);
		hostTimerDestroy(timer);
		inputImg.loadTimeMs = static_cast<float>(loadTime * 1000.0f);
		if (info) {
			printf("JFIF images read time = %.2f ms\n\n", loadTime * 1000.0);
		}
	}
	else {
		fprintf(stderr, "Input image file %s has not been found!\n", fname.c_str());
		return FAST_IO_ERROR;
	}

	return FAST_OK;
}

template<class Allocator>
fastStatus_t fvLoadBytestreams(char* folderName, std::list< Bytestream<Allocator> >& inputImg, bool info) {
	std::list<std::string> files;

	getFileList(folderName, files);
	if (files.empty()) {
		fprintf(stderr, "No input files found\n");
		return FAST_IO_ERROR;
	}

	hostTimer_t timer = NULL;
	if (info) {
		timer = hostTimerCreate();
		hostTimerStart(timer);
	}

	for (auto i = files.begin(); i != files.end(); ++i) {
		Bytestream<Allocator> bytestream;
		CHECK_FAST(fvLoadBytestream(*i, bytestream, false));
		inputImg.push_back(bytestream);
	}

	if (info) {
		const double loadTime = hostTimerEnd(timer);
		printf("JFIF images read time = %.2f ms\n\n", loadTime * 1000.0);
		hostTimerDestroy(timer);
	}

	return FAST_OK;
}

//////////////////////////////////////////////////////////////////////
/// 
//////////////////////////////////////////////////////////////////////
fastStatus_t fvSaveBytestream(
	std::string fname,
	unsigned char* inputImg,
	size_t size,
	bool info
);


template<class Allocator>
fastStatus_t fvSaveBytestream(std::string fname, Bytestream<Allocator>& inputImg, bool info) {
	return fvSaveBytestream(
		fname,
		inputImg.data.get(),
		inputImg.size,
		info
	);
}

template<class Allocator>
fastStatus_t fvSaveBytestream(char* folderName, std::list< Bytestream<Allocator> > images, bool info) {
	hostTimer_t timer = NULL;
	if (info) {
		timer = hostTimerCreate();
		hostTimerStart(timer);
	}

	int idx = 0;
	for (auto i = images.begin(); i != images.end(); ++i) {
		std::string fname(folderName);
		if (images.size() != 1) {
			const std::string::size_type loc = fname.find("*", 0);
			if (loc != std::string::npos) {
				std::ostringstream convert;
				convert << idx;
				fname.replace(loc, 1, convert.str());
			}
		}

		if (fvSaveBytestream(fname, *i, info) == FAST_OK) {
			idx++;
		}
	}

	if (info) {
		const double loadTime = hostTimerEnd(timer);
		printf("JFIF images write time = %.2f ms\n\n", loadTime * 1000.0);
		hostTimerDestroy(timer);
	}

	return FAST_OK;
}

#endif 
