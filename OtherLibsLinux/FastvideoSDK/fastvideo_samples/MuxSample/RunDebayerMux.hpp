/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __RUN_DEBAYER_MUX__
#define __RUN_DEBAYER_MUX__

#include "DebayerSampleOptions.h"

#include "Image.h"
#include "FastAllocator.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "helper_pfm.hpp"

#include "DebayerMux.h"

static fastStatus_t RunDebayerMux(DebayerSampleOptions &options) {
	DebayerMux hMux(options.Info);
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

	for (auto i = inputImg.begin(); i != inputImg.end(); ++i) {
		if ((*i).surfaceFmt != FAST_I8 && (*i).surfaceFmt != FAST_I12 && (*i).surfaceFmt != FAST_I16) {
			fprintf(stderr, "Input file must not be color\n");
			return FAST_IO_ERROR;
		}
		options.SurfaceFmt = (*i).surfaceFmt;
		options.BitsPerChannel = (*i).bitsPerChannel;
	}

	printf("Input surface format: grayscale\n");
	printf("Pattern: %s\n", EnumToString(options.Debayer.BayerFormat));
	printf("Output surface format: %s\n", EnumToString(BaseOptions::GetSurfaceFormatFromExtension(options.OutputPath)));
	printf("Debayer algorithm: %s\n", EnumToString(options.Debayer.BayerType));


	ImageT<float, FastAllocator> matrixA;
	if (options.GrayscaleCorrection.MatrixA != NULL) {
		unsigned channels;
		bool failed = false;

		printf("\nMatrix A: %s\n", options.GrayscaleCorrection.MatrixA);
		CHECK_FAST(fvLoadPFM(options.GrayscaleCorrection.MatrixA, matrixA.data, matrixA.w, matrixA.wPitch, FAST_ALIGNMENT, matrixA.h, channels));

		if (channels != 1) {
			fprintf(stderr, "Matrix A file must not be color\n");
			failed = true;
		}

		if (options.MaxHeight != matrixA.h || options.MaxWidth != matrixA.w) {
			fprintf(stderr, "Input and matrix A file parameters mismatch\n");
			failed = true;
		}

		if (failed) {
			fprintf(stderr, "Matrix A file reading error. Ignore parameters\n");
			failed = false;
		}
	}

	ImageT< char, FastAllocator > matrixB;
	if (options.GrayscaleCorrection.MatrixB != NULL) {
		bool failed = false;

		printf("\nMatrix B: %s\n", options.GrayscaleCorrection.MatrixB);
		CHECK_FAST(fvLoadImage(std::string(options.GrayscaleCorrection.MatrixB), std::string(""), matrixB, options.MaxHeight, options.MaxWidth, 8, false));

		if (matrixB.surfaceFmt != FAST_I8 &&  matrixB.surfaceFmt != FAST_I12 &&  matrixB.surfaceFmt != FAST_I16) {
			fprintf(stderr, "Matrix B file must not be color\n");
			failed = true;
		}

		if (options.MaxHeight != matrixB.h || options.MaxWidth != matrixB.w) {
			fprintf(stderr, "Input and matrix B file parameters mismatch\n");
			failed = true;
		}

		if (failed) {
			fprintf(stderr, "Matrix B file reading error. Ignore parameters\n");
		}
	}

	CHECK_FAST(hMux.Init(options, options.GrayscaleCorrection.MatrixA != NULL ? matrixA.data.get() : NULL, options.GrayscaleCorrection.MatrixB != NULL ? matrixB.data.get() : NULL));
	CHECK_FAST(hMux.Transform(inputImg));

	inputImg.clear();
	CHECK_FAST(hMux.Close());

	return FAST_OK;
}

#endif // __RUN_DEBAYER_MUX__