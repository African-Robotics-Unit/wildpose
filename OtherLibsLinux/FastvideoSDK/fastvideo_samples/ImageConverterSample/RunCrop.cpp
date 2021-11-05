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

#include "SurfaceTraits.hpp"
#include "supported_files.hpp"

#include "CropSampleOptions.h"

fastStatus_t RunCrop(const CropSampleOptions options) {
	Image<FastAllocator> inputImg;
	CHECK_FAST(fvLoadImage(options.InputPath, "", inputImg, 0, 0, 0, false));

	std::unique_ptr<unsigned char, FastAllocator> h_Result;
	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(h_Result.reset((unsigned char *)alloc.allocate(GetBufferSizeFromSurface(inputImg.surfaceFmt, options.Crop.CropWidth, options.Crop.CropHeight))));

	unsigned char *src = inputImg.data.get();
	unsigned char *dst = h_Result.get();

	src += options.Crop.CropLeftTopCoordsY * inputImg.wPitch +
		options.Crop.CropLeftTopCoordsX *
		GetBytesPerChannelFromSurface(inputImg.surfaceFmt) *
		GetNumberOfChannelsFromSurface(inputImg.surfaceFmt);
	
	const unsigned width = options.Crop.CropWidth *
		GetBytesPerChannelFromSurface(inputImg.surfaceFmt) *
		GetNumberOfChannelsFromSurface(inputImg.surfaceFmt);
	const unsigned pitch = GetPitchFromSurface(inputImg.surfaceFmt, options.Crop.CropWidth);
	for (unsigned y = 0; y < options.Crop.CropHeight; y++) {
		memcpy(&dst[y * pitch], &src[y * inputImg.wPitch], width);
	}
	
	CHECK_FAST_SAVE_FILE(fvSaveImageToFile(
		options.OutputPath,
		h_Result,
		inputImg.surfaceFmt,
		options.Crop.CropHeight,
		options.Crop.CropWidth,
		pitch,
		false
	));

	return FAST_OK;
}