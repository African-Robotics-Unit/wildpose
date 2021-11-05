/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <cstdio>

#include "FastAllocator.h"
#include "checks.h"
#include "Image.h"

#include "IdentifySurface.hpp"
#include "SurfaceTraits.hpp"
#include "supported_files.hpp"

#include "BaseOptions.h"

unsigned clamp(float value, unsigned bits) {
	if (value > float(1<<bits)) return 1 << bits;
	if (value < 0.f) return 0;
	return value;
}

fastStatus_t RunColorConvert(BaseOptions options) {
	Image<FastAllocator> inputImg;

	printf("Input image: %s\n", options.InputPath);
	printf("Output image: %s\n", options.OutputPath);

	CHECK_FAST(fvLoadImage(options.InputPath, "", inputImg, 0, 0, 0, false));

	const fastSurfaceFormat_t dstSurfaceFmt = IdentifySurface(GetBitsPerChannelFromSurface(inputImg.surfaceFmt), 1);

	std::unique_ptr<unsigned char, FastAllocator> h_Result;
	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(h_Result.reset((unsigned char *)alloc.allocate(GetBufferSizeFromSurface(dstSurfaceFmt, inputImg.w, inputImg.h))));
	const unsigned bits = GetBitsPerChannelFromSurface(inputImg.surfaceFmt);
	unsigned dstPitch = inputImg.w * GetBytesPerChannelFromSurface(dstSurfaceFmt);
	if (GetBytesPerChannelFromSurface(inputImg.surfaceFmt) == 1) {
		unsigned char *src = inputImg.data.get();
		unsigned char *dst = h_Result.get();

		for (unsigned y = 0; y < inputImg.h; y++) {
			for (unsigned x = 0; x < inputImg.w; x++) {
				const unsigned char R = src[y * inputImg.wPitch + x * 3];
				const unsigned char G = src[y * inputImg.wPitch + x * 3 + 1];
				const unsigned char B = src[y * inputImg.wPitch + x * 3 + 2];

				dst[y * dstPitch + x] = clamp(0.30f * R + 0.59f * G + 0.11f * B, bits );
			}
		}
	}
	else if (GetBytesPerChannelFromSurface(inputImg.surfaceFmt) == 2) {
		unsigned short *src = (unsigned short *)inputImg.data.get();
		unsigned short *dst = (unsigned short *)h_Result.get();
		unsigned pitch = inputImg.wPitch / sizeof(unsigned short);
		unsigned dstPitchWord = dstPitch / sizeof(unsigned short);
		for (unsigned y = 0; y < inputImg.h; y++) {
			for (unsigned x = 0; x < inputImg.w; x++) {
				const unsigned short R = src[y * pitch + x * 3];
				const unsigned short G = src[y * pitch + x * 3 + 1];
				const unsigned short B = src[y * pitch + x * 3 + 2];

				dst[y * dstPitchWord + x] = clamp(0.30f * R + 0.59f * G + 0.11f * B, bits);
			}
		}
	}

	return fvSaveImageToFile(
		options.OutputPath,
		h_Result, dstSurfaceFmt,
		inputImg.h, inputImg.w, dstPitch,
		false
	);
}