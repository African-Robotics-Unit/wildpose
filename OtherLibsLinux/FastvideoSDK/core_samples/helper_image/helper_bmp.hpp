/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef FVHELPER_BMP_H
#define FVHELPER_BMP_H

#include "fastvideo_sdk.h"
#include "bmp.hpp"
#include "SurfaceTraits.hpp"

template <class T, class Allocator> fastStatus_t
fvLoadBMP(
	std::unique_ptr<T, Allocator> &data,
	unsigned            &surfaceHeight,
	unsigned            &surfaceWidth,
	unsigned            &surfacePitch8,
	fastSurfaceFormat_t &surfaceFmt,
	const char *filename
) {
	Allocator alloc;
	void *p = NULL;
	unsigned channels;
	const bmpStatus_t ret = LoadBMP(&p, &alloc, surfaceHeight, surfaceWidth, surfacePitch8, channels, filename);
	if (ret != BMP_OK) {
		switch (ret) {
			case BMP_INVALID_FORMAT:
				return FAST_INVALID_FORMAT;
			case BMP_IO_ERROR:
				return  FAST_IO_ERROR;
			case BMP_UNSUPPORTED_FORMAT:
				return  FAST_UNSUPPORTED_FORMAT;
			default:
				return FAST_IO_ERROR;
		}
	}

	data.reset(static_cast<T*>(p));
	surfaceFmt = (channels == 1) ? FAST_I8 : FAST_BGR8;

	return FAST_OK;
}

template <class T, class Allocator> fastStatus_t
fvStoreBMP(
	const char						*filename,
	std::unique_ptr<T, Allocator>	&data,
	fastSurfaceFormat_t				surfaceFmt,
	unsigned						surfaceHeight,
	unsigned						surfaceWidth,
	unsigned						surfacePitch8
) {
	unsigned char *p = (unsigned char *)data.get();
	const int nChannels = GetNumberOfChannelsFromSurface(surfaceFmt);

	if (StoreBMP(filename, (unsigned char *)data.get(), nChannels, surfaceHeight, surfaceWidth, surfacePitch8) != BMP_OK)
		return FAST_IO_ERROR;

	return FAST_OK;
}

#endif //FVHELPER_BMP_H