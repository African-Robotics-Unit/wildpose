/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "RunNppSamples.hpp"
#include "Resize2.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "checks.h"
#include "ResizeTypeToStr.h"

fastStatus_t RunResize(NppResizeSampleOptions &options) {
	Resize hResize(options.Info);

	std::list< Image<FastAllocator> > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, 12, false));
	} else {
		Image<FastAllocator> img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 12, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	options.SurfaceFmt = (*inputImg.begin()).surfaceFmt;

	// correct destination buffer height
	if (!options.Resize.ResizedHeightEnabled) {
		const double correctionCoefficient = 1.01;
		double maxScaleFactor = double(options.MaxWidth) / double(options.Resize.ResizedWidth);
		options.Resize.ResizedHeight = static_cast<unsigned>(correctionCoefficient * options.MaxHeight / maxScaleFactor);
	}

	printf("Input surface format: %s\n", EnumToString(options.SurfaceFmt));
	printf("Destination buffer size: %d x %d\n", options.Resize.ResizedWidth, options.Resize.ResizedHeight);
	printf("Shift: %.3f\n", options.Resize.Shift);
	printf("Resize type: %s\n", ResizeTypeToStr(options.Interpolation.Type));

	CHECK_FAST(hResize.Init(options));
	CHECK_FAST(hResize.Transform(inputImg));

	inputImg.clear();
	CHECK_FAST(hResize.Close());

	return FAST_OK;
}
