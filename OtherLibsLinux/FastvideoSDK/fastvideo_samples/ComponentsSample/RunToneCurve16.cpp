/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "RunToneCurve16.hpp"
#include "ToneCurve16.h"

#include "Image.h"
#include "FastAllocator.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"

fastStatus_t RunToneCurve(ToneCurveSampleOptions &options) {
	std::list< Image<FastAllocator> > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, 16, false));
	} else {
		Image<FastAllocator> img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 16, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	for (auto i = inputImg.begin(); i != inputImg.end(); i++) {
		if ((*i).surfaceFmt != FAST_RGB16) {
			fprintf(stderr, "Unsupported surface format. Just 16-bit images supported\n");
			return FAST_IO_ERROR;
		}
	}
	options.SurfaceFmt = (*inputImg.begin()).surfaceFmt;

	printf("Input surface format: %s\n", EnumToString(options.SurfaceFmt));

	std::unique_ptr<unsigned short, FastAllocator> toneCurve;
	{
		FastAllocator alloc;
		toneCurve.reset((unsigned short *)alloc.allocate(1024 * sizeof(unsigned short)));

		const float coeff = (1 << 16) - 1;

		FILE *fp = fopen(options.ToneCurve.ToneCurveFile, "r");
		if (!fp) {
			fprintf(stderr, "Tone curve file loading error: %s\n", options.ToneCurve.ToneCurveFile);
			return FAST_IO_ERROR;
		}

		for (int i = 0; i < 1024; i++) {
			float value = 0;
			int filled = fscanf(fp, "%f", &value);
			if (filled != 1) {
				fprintf(stderr, "Incorrect tone curve file: must be 1024 elements (%d elements was readed)\n", i);
				return FAST_IO_ERROR;
			}

			toneCurve.get()[i] = value * coeff;
		}

		fclose(fp);
	}
	
	ToneCurve16 hToneCurve(options.Info);
	CHECK_FAST(hToneCurve.Init(options, toneCurve.get()));
	CHECK_FAST(hToneCurve.Transform(inputImg, NULL));

	inputImg.clear();
	CHECK_FAST(hToneCurve.Close());

	return FAST_OK;
}
