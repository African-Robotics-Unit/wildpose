/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#ifndef __RUN_SHARPEN__
#define __RUN_SHARPEN__

#include <cstdio>

#include "Sharpen.h"

#include "ImageFilterSampleOptions.h"

#include "checks.h"
#include "supported_files.hpp"
#include "FastAllocator.h"

static fastStatus_t RunSharpen(ImageFilterSampleOptions &options, bool IsSharpenFilter) {
	Sharp hSharp(options.Info);
	std::list< Image<FastAllocator > > inputImg;

	fastStatus_t ret;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, options.MaxWidth, options.MaxHeight, 8, false));
	} else {
		Image< FastAllocator > img;

		ret = fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, options.MaxWidth, options.MaxHeight, 8, false);
		if (ret == FAST_IO_ERROR) {
			fprintf(stderr, "Input image file %s has not been found!\n", options.InputPath);
			return ret;
		}
		if (ret != FAST_OK)
			return ret;

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	if (GetBitsPerChannelFromSurface((*inputImg.begin()).surfaceFmt) != 8) {
		fprintf(stderr, "Unsupported surface format: only 8-bits images are supported\n");
		return FAST_IO_ERROR;
	}

	options.SurfaceFmt = (*inputImg.begin()).surfaceFmt;

	CHECK_FAST(hSharp.Init(options, IsSharpenFilter));
	CHECK_FAST(hSharp.Transform(inputImg, options.ImageFilter.Sigma));
	CHECK_FAST(hSharp.Close());

	return FAST_OK;
}

#endif // __RUN_SHARPEN__