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

#include "fastvideo_sdk.h"
#include "FastAllocator.h"

#include "checks.h"
#include "supported_files.hpp"

#include "fastvideo_encoder_j2k.h"

#include "J2kEncoderOptions.h"

#include "J2kEncoderOneImage.h"
#include "J2kEncoderBatch.h"

fastStatus_t RunJ2kEncoder(J2kEncoderOptions options) {
	fastStatus_t ret = FAST_OK;

	if (options.Algorithm == FAST_ENCODER_J2K_ALGORITHM_UNKNOWN) {
		fprintf(stderr, "Unknown algorithm.\n");
		return FAST_INVALID_VALUE;
	}

	if (options.Algorithm != FAST_ENCODER_J2K_ALGORITHM_ENCODE_REVERSIBLE && options.Algorithm != FAST_ENCODER_J2K_ALGORITHM_ENCODE_IRREVERSIBLE) {
		fprintf(stderr, "Sorry, this algorithm haven't been implemented yet.\n");
		return FAST_UNAPPLICABLE_OPERATION;
	}
	
	fastSdkParametersHandle_t hSdkParameters;
	{
		CHECK_FAST(fastGetSdkParametersHandle(&hSdkParameters));
		CHECK_FAST(fastEncoderJ2kLibraryInit(hSdkParameters));
	}

	std::list< Image<FastAllocator> > inputImages;
	Image<FastAllocator> inputImage;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImages, 0, 0, 8, false));
	} else {
		ret = fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath),
			inputImage, 0, 0, 8, options.Info);
		if (ret != FAST_OK) {
			if (ret == FAST_IO_ERROR) fprintf(stderr, "Input image file %s has not been found!\n", options.InputPath);
			return ret;
		}
		options.MaxHeight = options.MaxHeight == 0 ? inputImage.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? inputImage.w : options.MaxWidth;
		inputImages.push_back(inputImage);
	}
	options.SurfaceFmt = inputImages.begin()->surfaceFmt;

	if (options.BatchSize > inputImages.size() * options.RepeatCount)
		options.RepeatCount = options.BatchSize;

	if (options.BatchSize == 1) {
		J2kEncoderOneImage encoder(options.Info);
		ret = encoder.Init(options);
		if (ret != FAST_OK)
		{
			if (ret == FAST_INSUFFICIENT_DEVICE_MEMORY)
				fprintf(stderr, "Insufficient device memory.\n");
			return ret;
		}
		CHECK_FAST(encoder.Transform(inputImages));
		CHECK_FAST(encoder.Close());
	} else {
		J2kEncoderBatch encoder(options.Info);
		ret = encoder.Init(options);
		if (ret != FAST_OK)
		{
			if (ret == FAST_INSUFFICIENT_DEVICE_MEMORY)
				fprintf(stderr, "Insufficient device memory.\n");
			return ret;
		}
		CHECK_FAST(encoder.Transform(inputImages));
		CHECK_FAST(encoder.Close());
	}

	inputImages.clear();

	return ret;
}
