/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __HELPER_PPM__
#define __HELPER_PPM__

#include "fastvideo_sdk.h"
#include "ppm.h"
#include "SurfaceTraits.hpp"
#include "IdentifySurface.hpp"
#include <memory>

template <class T, class Allocator> fastStatus_t
fvLoadPPM(
	const char *file, std::unique_ptr<T, Allocator> &data, 
	unsigned int &w, unsigned int &wPitch, unsigned int &h, 
	fastSurfaceFormat_t &surfaceFmt
) {
	Allocator alloc;
	void *p = NULL;
	unsigned channels = 0, bitsPerChannel = 0;
	if (!LoadPPM(file, &p, &alloc, w, wPitch, h, bitsPerChannel, channels))
		return FAST_IO_ERROR;
	data.reset((T*)p);
	surfaceFmt = IdentifySurface(bitsPerChannel, channels);
	return FAST_OK;
}

template <class T, class Allocator> fastStatus_t
fvSavePPM(const char *file, std::unique_ptr<T, Allocator> &data, fastSurfaceFormat_t surfaceFmt, unsigned int w, unsigned wPitch, unsigned int h) {
	unsigned char *p = (unsigned char *)data.get();
	const int bitsPerChannel = GetBitsPerChannelFromSurface(surfaceFmt);
	const int nChannels = GetNumberOfChannelsFromSurface(surfaceFmt);

	if (!SavePPM(file, (unsigned char *)data.get(), w, wPitch, h, bitsPerChannel, nChannels))
		return FAST_IO_ERROR;
	return FAST_OK;
}

#endif //
