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

#include "fastvideo_sdk.h"
#include "tiff.h"
#include "SurfaceTraits.hpp"
#include "IdentifySurface.hpp"
#include <memory>

template <class T, class Allocator> fastStatus_t
fvLoadTIFF(	const char* file, ImageT<T, Allocator>& image) {
	Allocator alloc;
	void* p = NULL;
	unsigned channels = 0, bitsPerChannel = 0;
		
	if (!LoadTIFF(file, &p, &alloc, image.w, image.wPitch, image.h, bitsPerChannel, channels))
		return FAST_IO_ERROR;
	image.data.reset((T*)p);

	image.surfaceFmt = IdentifySurface(bitsPerChannel, channels);

	return FAST_OK;
}

template <class T, class Allocator> fastStatus_t
fvSaveTIFF(const char* file, Image<Allocator>& image) {
	const int bytesPerChannel = GetBytesPerChannelFromSurface(image.surfaceFmt);
	const int nChannels = GetNumberOfChannelsFromSurface(image.surfaceFmt);

	if (!saveTIFF(file, (void*)image.data.get(), image.w, image.wPitch * sizeof(T), image.h, bytesPerChannel, nChannels))
		return FAST_IO_ERROR;
	return FAST_OK;
}

template <class T, class Allocator> fastStatus_t
fvSaveTIFF(const char* file, std::unique_ptr<T, Allocator>& data, fastSurfaceFormat_t surfaceFmt, unsigned int w, unsigned pitch, unsigned int h) {
	const int bytesPerChannel = GetBytesPerChannelFromSurface(surfaceFmt);
	const int nChannels = GetNumberOfChannelsFromSurface(surfaceFmt);

	if (!SaveTIFF(file, (void*)data.get(), w, pitch, h, bytesPerChannel, nChannels))
		return FAST_IO_ERROR;
	return FAST_OK;
}
