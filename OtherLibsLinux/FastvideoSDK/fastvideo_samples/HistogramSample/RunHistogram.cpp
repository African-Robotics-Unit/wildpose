/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "Histogram.h"

#include "FastAllocator.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "checks.h"

const char *AnalyzeHistogram(const fastHistogramType_t type) {
	switch(type) {
		case FAST_HISTOGRAM_COMMON:
			return "common";
		case FAST_HISTOGRAM_PARADE:
			return "parade";
		case FAST_HISTOGRAM_BAYER:
			return "bayer";
		case FAST_HISTOGRAM_BAYER_G1G2:
			return "bayer G1G2";
	}
	return "unknown";
}

fastStatus_t RunHistogram(HistogramSampleOptions &options) {
	std::list< Image<FastAllocator> > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, options.BitsPerChannel, false));
	} else {
		Image<FastAllocator> img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 8, false));
		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		options.Histogram.RoiWidth = options.Histogram.RoiWidth == 0 ? (img.w - options.Histogram.RoiLeftTopX) : options.Histogram.RoiWidth;
		options.Histogram.RoiHeight = options.Histogram.RoiHeight == 0 ? (img.h - options.Histogram.RoiLeftTopY) : options.Histogram.RoiHeight;
		inputImg.push_back(img);
	}
	options.BitsPerChannel = (*inputImg.begin()).bitsPerChannel;
	options.SurfaceFmt = (*inputImg.begin()).surfaceFmt;

	printf("Input surface format: %s\n", EnumToString((*inputImg.begin()).surfaceFmt));
	printf("Histogram type: %s\n", AnalyzeHistogram(options.Histogram.HistogramType));

	if (static_cast<unsigned>(options.Histogram.BinCount) > (1U << GetBitsPerChannelFromSurface(options.SurfaceFmt))) {
		fprintf(stderr, "Incorrect number of bins was provided for histogram calculation of %d-bit image.", GetBitsPerChannelFromSurface(options.SurfaceFmt));
		return FAST_INVALID_SIZE;
	}
	if (options.Histogram.HistogramType == FAST_HISTOGRAM_PARADE && options.SurfaceFmt != FAST_RGB8) {
		fprintf(stderr, "Parade histogram may be calculated only for 8-bit RGB images.");
		return FAST_INVALID_FORMAT;
	}

	if ((options.Histogram.HistogramType == FAST_HISTOGRAM_BAYER ||
		 options.Histogram.HistogramType == FAST_HISTOGRAM_BAYER_G1G2) &&
		GetNumberOfChannelsFromSurface(options.SurfaceFmt) != 1) {
		fprintf(stderr, "This type of histogram may be calculated only for Bayer (8-bit gray) images.");
		return FAST_INVALID_FORMAT;
	}

	Histogram histogram;
	CHECK_FAST(histogram.Init(options));
	CHECK_FAST(histogram.Calculate(inputImg));
	inputImg.clear();
	CHECK_FAST(histogram.Close());

	return FAST_OK;
}
