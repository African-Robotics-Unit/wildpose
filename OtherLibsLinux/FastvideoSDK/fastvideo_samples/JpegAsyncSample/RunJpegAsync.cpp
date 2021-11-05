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

#include "JpegEncoderSampleOptions.h"
#include "JpegEncoderAsync.h"

#include "Image.h"
#include "checks.h"
#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "FastAllocator.h"

fastStatus_t RunJpegAsync(JpegEncoderSampleOptions &options) {
	std::list< Image<FastAllocator> > inputImgs;

	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImgs, 0, 0, 0, false));
	} else {
		Image<FastAllocator> img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, options.MaxHeight, options.MaxWidth, 8, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImgs.push_back(img);
	}

	for (auto i = inputImgs.begin(); i != inputImgs.end(); i++) {
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

	printf("Surface format: %s\n", EnumToString((*(inputImgs.begin())).surfaceFmt));
	printf("Sampling format: %s\n", EnumToString((*(inputImgs.begin())).samplingFmt));
	printf("JPEG quality: %d%%\n", options.JpegEncoder.Quality);
	printf("Restart interval: %d\n", options.JpegEncoder.RestartInterval);

	JpegEncoderAsync hEncoder(options.Info);
	CHECK_FAST(hEncoder.Init(options));
	CHECK_FAST(hEncoder.Encode(inputImgs));

	inputImgs.clear();
	CHECK_FAST(hEncoder.Close());

	return FAST_OK;
}
