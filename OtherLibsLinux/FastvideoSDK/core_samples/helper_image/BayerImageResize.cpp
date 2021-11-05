/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "fastvideo_sdk.h"

#include <cstdio>

#include "SurfaceTraits.hpp"
#include "FastAllocator.h"
#include "Image.h"

int BayerMergeLines(Image<FastAllocator> &img) {
	const unsigned srcWidth = img.GetBytesPerPixel() * img.w;

	const unsigned dstWidth = img.w * 2;
	const unsigned dstPitch = GetPitchFromSurface(img.surfaceFmt, dstWidth);
	const unsigned dstHeight = img.h / 2;

	if (srcWidth != img.wPitch) {
		FastAllocator alloc;
		unsigned char *tmp = (unsigned char*)alloc.allocate(dstPitch * dstHeight);

		for (unsigned srcY = 0, dstY = 0; srcY < img.h; srcY += 2, dstY++) {
			memcpy(
				&tmp           [dstY * dstPitch],
				&img.data.get()[srcY * img.wPitch],
				srcWidth
			);
			memcpy(
				&tmp           [ dstY      * dstPitch + srcWidth],
				&img.data.get()[(srcY + 1) * img.wPitch],
				srcWidth
			);
		}

		img.data.reset(tmp);
	}

	img.w = dstWidth;
	img.wPitch = dstPitch;
	img.h = dstHeight;
	return 0;
}

int BayerSplitLines(Image<FastAllocator> &img) {
	const unsigned srcWidth = img.GetBytesPerPixel() * img.w;

	const unsigned dstWidth = img.w / 2;
	const unsigned dstPitch = GetPitchFromSurface(img.surfaceFmt, dstWidth);
	const unsigned dstHeight = img.h * 2;
	const unsigned dstWidthInBytes = dstWidth * img.GetBytesPerPixel();

	if ((dstWidth * GetBytesPerChannelFromSurface(img.surfaceFmt)) != dstPitch) {
		FastAllocator alloc;
		unsigned char *tmp = (unsigned char*)alloc.allocate(dstPitch * dstHeight);

		for (unsigned srcY = 0, dstY = 0; srcY < img.h; srcY++, dstY += 2) {
			memcpy(
				&tmp[dstY * dstPitch],
				&img.data.get()[srcY * img.wPitch],
				dstWidthInBytes
			);
			memcpy(
				&tmp[(dstY + 1) * dstPitch],
				&img.data.get()[srcY * img.wPitch + dstWidthInBytes],
				dstWidthInBytes
			);
		}

		img.data.reset(tmp);
	}

	img.w = dstWidth;
	img.wPitch = dstPitch;
	img.h = dstHeight;
	return 0;
}
