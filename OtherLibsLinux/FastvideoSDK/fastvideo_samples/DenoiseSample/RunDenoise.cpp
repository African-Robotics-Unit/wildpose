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
#include <list>

#include "RunDenoise.hpp"

#include "Denoise.h"
#include "supported_files.hpp"
#include "checks.h"

fastStatus_t RunDenoise(DenoiseOptions options) {
	std::list< Image<FastAllocator> > inputImages;
	Image<FastAllocator> inputImage;

	if (options.Function == FAST_THRESHOLD_FUNCTION_UNKNOWN) {
		fprintf(stderr, "Unknown thresholding function.\n");
		return FAST_INVALID_VALUE;
	}

	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImages, 0, 0, 8, false));
	} else {
		fastStatus_t ret = fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath),
			inputImage, 0, 0, 8, options.Info);
		if (ret != FAST_OK) {
			if (ret == FAST_IO_ERROR) fprintf(stderr, "Input image file %s has not been found!\n", options.InputPath);
			return ret;
		}
		options.MaxHeight = options.MaxHeight == 0 ? inputImage.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? inputImage.w : options.MaxWidth;
		inputImages.push_back(inputImage);
		options.SurfaceFmt = inputImage.surfaceFmt;
	}

	Denoise denoise(options.Info);
	CHECK_FAST(denoise.Init(options));
	CHECK_FAST(denoise.Transform(inputImages));
	CHECK_FAST(denoise.Close());
	inputImages.clear();

	return FAST_OK;
}
