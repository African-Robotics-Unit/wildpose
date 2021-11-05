/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __RUN_MEDIAN_FILTER__
#define __RUN_MEDIAN_FILTER__

#include "RunMedian.hpp"
#include "Median.h"

#include "Image.h"
#include "FastAllocator.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "SurfaceTraits.hpp"

fastStatus_t RunMedianFilter(BaseOptions &options) {
	std::list< Image<FastAllocator > > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, 8, false));
	} else {
		Image<FastAllocator> img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 8, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	options.SurfaceFmt = (*inputImg.begin()).surfaceFmt;
	if (GetBytesPerChannelFromSurface(options.SurfaceFmt) == 1) {
		fprintf(stderr, "Unsupported surface format\n");
		return FAST_UNSUPPORTED_SURFACE;
	}

	printf("Input surface format: %s\n", EnumToString((*inputImg.begin()).surfaceFmt));
	printf("Output surface format: %s\n", EnumToString(options.SurfaceFmt));

	Median hMedian(options.Info);
	CHECK_FAST(hMedian.Init(options));
	CHECK_FAST(hMedian.Transform(inputImg));

	inputImg.clear();
	CHECK_FAST(hMedian.Close());

	return FAST_OK;
}

#endif // __RUN_MEDIAN_FILTER__