/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include <list>

#include "Encoder.h"

#include "JpegEncoderSampleOptions.h"

#include "Image.h"
#include "checks.h"
#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "FastAllocator.h"
#include "helper_quant_table.hpp"
#include "BayerImageResize.h"

fastStatus_t RunJpegEncoder(JpegEncoderSampleOptions &options) {
	Encoder hEncoder(options.Info, true, false);
	std::list<Image<FastAllocator> > inputImgs;

	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImgs, 0, 0, 0, false));
	} else {
		Image<FastAllocator> img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, options.MaxHeight, options.MaxWidth, 8, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImgs.push_back(img);
	}

	for (auto i = inputImgs.begin(); i != inputImgs.end(); ++i) {
		// correct options if grayscale image
		switch ((*i).surfaceFmt) {
			case FAST_I8:
			{
				if (options.JpegEncoder.GrayAsRGB) {
					(*i).samplingFmt = FAST_JPEG_420;
					options.JpegEncoder.SamplingFmt = (*i).samplingFmt;
					options.SurfaceFmt = FAST_I8;
				} else {
					(*i).samplingFmt = FAST_JPEG_Y;
					options.JpegEncoder.SamplingFmt = (*i).samplingFmt;
					options.SurfaceFmt = FAST_I8;
				}
			}
			break;

			case FAST_I12:
			{
				(*i).samplingFmt = options.JpegEncoder.SamplingFmt = FAST_JPEG_Y;
				options.SurfaceFmt = FAST_I12;
			}
			break;

			default:
			{
				(*i).samplingFmt = options.JpegEncoder.SamplingFmt;
				options.SurfaceFmt = (*i).surfaceFmt;
			}
			break;
		}
	}

	if (options.JpegEncoder.BayerCompression) {
		for (auto i = inputImgs.begin(); i != inputImgs.end(); ++i) {
			if (((*i).h & 1) == 0) {
				BayerMergeLines(*i);
			} else {
				fprintf(stderr, "Bayer compression can not be applied for odd image height\n");
				continue;
			}

			options.MaxHeight = std::max(options.MaxHeight,(*i).h);
			options.MaxWidth = std::max(options.MaxWidth, (*i).w);
		}
	}

	printf("Surface format: %s\n", EnumToString((*(inputImgs.begin())).surfaceFmt));
	printf("Sampling format: %s\n", EnumToString((*(inputImgs.begin())).samplingFmt));
	printf("JPEG quality: %d%%\n", options.JpegEncoder.Quality);

	int restartInterval = 32;
	if ((*(inputImgs.begin())).surfaceFmt == FAST_RGB8 || (*(inputImgs.begin())).surfaceFmt == FAST_RGB12) {
		if ((*(inputImgs.begin())).samplingFmt == FAST_JPEG_444 )
			restartInterval = 10;
		if ((*(inputImgs.begin())).samplingFmt == FAST_JPEG_422)
			restartInterval = 8;
		if ((*(inputImgs.begin())).samplingFmt == FAST_JPEG_420)
			restartInterval = 5;
	}

	printf("Restart interval: %d\n", restartInterval);

	fastJpegQuantState_t quantState = { 0 };
	bool quantStateEnabled = false;
	if (options.JpegEncoder.QuantTableFileName != NULL) {
		printf("External quant table: %s\n", options.JpegEncoder.QuantTableFileName);
		CHECK_FAST(fvLoadQuantTable(options.JpegEncoder.QuantTableFileName, quantState));
		quantStateEnabled = true;
	}

	CHECK_FAST(hEncoder.Init(options, NULL));
	CHECK_FAST(hEncoder.Encode(inputImgs, quantStateEnabled ? &quantState : NULL, 0, NULL));

	inputImgs.clear();
	CHECK_FAST(hEncoder.Close());

	return FAST_OK;
}
