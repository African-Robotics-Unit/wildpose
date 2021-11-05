/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "supported_files.hpp"
#include "Resize.h"
#include "ResizerSampleOptions.h"

#include "FastAllocator.h"

#include "ResizeHelper.h"

fastStatus_t RunResizer(ResizerSampleOptions options) {
	std::list< Image<FastAllocator > > inputImg;

	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, 0, false));
		int idx = 0;
		for (auto i = inputImg.begin(); i != inputImg.end(); i++, idx++) {
			i->outputFileName = generateOutputFileName(options.OutputPath, idx);
			options.SurfaceFmt = i->surfaceFmt;
		}
	} else {
		Image<FastAllocator > img;

		fastStatus_t ret = fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, options.MaxHeight, options.MaxWidth, 8, false);
		if (ret == FAST_IO_ERROR) {
			fprintf(stderr, "Input image file %s has not been found!\n", options.InputPath);
			return ret;
		}
		if (ret != FAST_OK)
			return ret;

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		img.outputFileName = std::string(options.OutputPath);
		inputImg.push_back(img);
		options.SurfaceFmt = img.surfaceFmt;
	}
	
	double maxScaleFactor = GetResizeMaxScaleFactor(options.MaxWidth, options.MaxHeight, options.Resize);
	if (!options.Resize.OutputHeightEnabled) {
		options.Resize.OutputHeight = GetResizeMaxHeight(options.MaxHeight, maxScaleFactor);
	}

	if (maxScaleFactor > ResizerOptions::SCALE_FACTOR_MAX) {
		fprintf(stderr, "Incorrect image scale factor (%.3f). Max scale factor is %d\n", maxScaleFactor, ResizerOptions::SCALE_FACTOR_MAX);
		return FAST_INVALID_VALUE;
	}

	if (options.IsFolder) {
		printf("Maximum scale factor: %.3f\n", maxScaleFactor);
	}

	Resizer hResizer(options.Info);
	CHECK_FAST(hResizer.Init(options, maxScaleFactor));
	CHECK_FAST(hResizer.Resize(inputImg, options.OutputPath));
	CHECK_FAST(hResizer.Close());

	return FAST_OK;
}
