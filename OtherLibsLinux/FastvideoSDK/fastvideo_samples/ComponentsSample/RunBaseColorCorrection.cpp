/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "RunBaseColorCorrection.hpp"
#include "BaseColorCorrection.h"

#include "Image.h"
#include "FastAllocator.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"

fastStatus_t RunBaseColorCorrection(BaseColorCorrectionSampleOptions &options) {
	std::list< Image<FastAllocator> > inputImg;
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

	printf("Input surface format: %s\n", EnumToString((*inputImg.begin()).surfaceFmt));
	printf("Output surface format: %s\n", EnumToString(options.SurfaceFmt));

	if (!options.BaseColorCorrection.BaseColorCorrectionEnabled) {
		return FAST_INVALID_VALUE;
	}

	printf("Correction matrix:\n");
	printf("\t%.3f\t%.3f\t%.3f\t%.3f\n", options.BaseColorCorrection.BaseColorCorrection[0], options.BaseColorCorrection.BaseColorCorrection[1], options.BaseColorCorrection.BaseColorCorrection[2], options.BaseColorCorrection.BaseColorCorrection[3]);
	printf("\t%.3f\t%.3f\t%.3f\t%.3f\n", options.BaseColorCorrection.BaseColorCorrection[4], options.BaseColorCorrection.BaseColorCorrection[5], options.BaseColorCorrection.BaseColorCorrection[6], options.BaseColorCorrection.BaseColorCorrection[7]);
	printf("\t%.3f\t%.3f\t%.3f\t%.3f\n", options.BaseColorCorrection.BaseColorCorrection[8], options.BaseColorCorrection.BaseColorCorrection[9], options.BaseColorCorrection.BaseColorCorrection[10], options.BaseColorCorrection.BaseColorCorrection[11]);

	printf("Min values (RGB): {%d, %d, %d}\n", options.BaseColorCorrection.WhiteLevel[0], options.BaseColorCorrection.WhiteLevel[1], options.BaseColorCorrection.WhiteLevel[2]);

	BaseColorCorrection hBaseColorCorrection(options.Info);
	CHECK_FAST(hBaseColorCorrection.Init(options));
	CHECK_FAST(hBaseColorCorrection.Transform(inputImg));

	inputImg.clear();
	CHECK_FAST(hBaseColorCorrection.Close());

	return FAST_OK;
}
