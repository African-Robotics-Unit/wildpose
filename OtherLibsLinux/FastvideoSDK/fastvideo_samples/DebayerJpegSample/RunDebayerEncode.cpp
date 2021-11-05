/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <cassert>

#include "supported_files.hpp"
#include "EnumToStringSdk.h"

#include "helper_pfm.hpp"
#include "helper_lut.hpp"

#include "Image.h"
#include "DebayerJpeg.h"
#include "DebayerJpegSampleOptions.h"

#include "FastAllocator.h"

fastStatus_t RunDebayerEncode(DebayerJpegSampleOptions options) {
	std::list< Image<FastAllocator> > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, options.RawWidth, options.RawHeight, options.BitsPerChannel, false));
	} else {
		Image<FastAllocator> img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, options.RawHeight, options.RawWidth, options.BitsPerChannel, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	for (auto i = inputImg.begin(); i != inputImg.end(); ++i) {
		if ((*i).surfaceFmt != FAST_I8 && (*i).surfaceFmt != FAST_I12) {
			fprintf(stderr, "Unsupported image format (just 8/12 bits grayscale images)\n");
			return FAST_UNSUPPORTED_FORMAT;
		}
		(*i).samplingFmt = options.JpegEncoder.SamplingFmt;
	}
	options.SurfaceFmt = inputImg.begin()->surfaceFmt;

	printf("Input surface format: grayscale\n");
	printf("Pattern: %s\n", EnumToString(options.Debayer.BayerFormat));
	printf("Output surface format: %s\n", EnumToString(options.SurfaceFmt));
	printf("Output sampling format: %s\n", EnumToString(options.JpegEncoder.SamplingFmt));
	printf("Debayer algorithm: %s\n", EnumToString(options.Debayer.BayerType));
	printf("JPEG quality: %d%%\n", options.JpegEncoder.Quality);
	printf("Restart interval: %d\n", options.JpegEncoder.RestartInterval);

	DebayerJpeg hDebayerJpeg(options.Info);
	CHECK_FAST(hDebayerJpeg.Init(options));
	CHECK_FAST(hDebayerJpeg.Transform(inputImg));

	inputImg.clear();
	CHECK_FAST(hDebayerJpeg.Close());

	return FAST_OK;
}
