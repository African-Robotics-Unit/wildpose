/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <iostream>

#include "supported_files.hpp"
#include "helper_pfm.hpp"

#include "Image.h"
#include "Sam.h"
#include "DebayerSampleOptions.h"

#include "FastAllocator.h"

bool is2BytesGrayscale(fastSurfaceFormat_t format) {
	return format == FAST_I12 || format == FAST_I16 || format == FAST_I10 || format == FAST_I14;
}

bool isGrayscale(fastSurfaceFormat_t format) {
	return format == FAST_I8 || is2BytesGrayscale(format);
}

fastStatus_t RunSam(DebayerSampleOptions &options, bool isTwoBytesMatrixB) {
	std::list<Image<FastAllocator>> inputImg;

	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, options.RawWidth, options.RawHeight, options.BitsPerChannel, false));
	} else {
		Image<FastAllocator > img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, options.RawHeight, options.RawWidth, options.BitsPerChannel, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	for (auto i = inputImg.begin(); i != inputImg.end(); ++i) {
		if (!isTwoBytesMatrixB) {
			if (!isGrayscale((*i).surfaceFmt)) {
				fprintf(stderr, "Input file must be gray-scale \n");
				return FAST_IO_ERROR;
			}
		} else {
			if (!is2BytesGrayscale((*i).surfaceFmt)) {
				fprintf(stderr, "Input file must be two byte gray-scale\n");
				return FAST_IO_ERROR;
			}
		}

		options.SurfaceFmt = (*i).surfaceFmt;
		options.BitsPerChannel = (*i).bitsPerChannel;
	}

	printf("Input surface format: grayscale\n");
	printf("Output surface format: grayscale\n");

	ImageT<float, FastAllocator> matrixA;
	if (options.GrayscaleCorrection.MatrixA != NULL) {
		unsigned channels;
		bool failed = false;

		printf("\nMatrix A: %s\n", options.GrayscaleCorrection.MatrixA);
		CHECK_FAST(fvLoadPFM(options.GrayscaleCorrection.MatrixA, matrixA.data, matrixA.w, matrixA.wPitch, FAST_ALIGNMENT, matrixA.h, channels));

		if (channels != 1) {
			fprintf(stderr, "Matrix A file must be gray-scale \n");
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

	ImageT< char, FastAllocator > matrixB8;
	ImageT< short, FastAllocator > matrixB16;
	if (options.GrayscaleCorrection.MatrixB != NULL) {
		bool failed = false;

		printf("\nMatrix B: %s\n", options.GrayscaleCorrection.MatrixB);


		if (!isTwoBytesMatrixB) {
			CHECK_FAST(fvLoadImage(std::string(options.GrayscaleCorrection.MatrixB), std::string(""), matrixB8, options.MaxHeight, options.MaxWidth, 8, false));

			if (matrixB8.surfaceFmt != FAST_I8) {
				fprintf(stderr, "Matrix B file must be 8-bits gray-scale\n");
				failed = true;
			}
			if (options.MaxHeight != matrixB8.h || options.MaxWidth != matrixB8.w) {
				fprintf(stderr, "Input and matrix B file parameters mismatch\n");
				failed = true;
			}
		} else {
			CHECK_FAST(fvLoadImage(std::string(options.GrayscaleCorrection.MatrixB), std::string(""), matrixB16, options.MaxHeight, options.MaxWidth, 8, false));
			if (!is2BytesGrayscale(matrixB16.surfaceFmt)) {
				fprintf(stderr, "Matrix B file must be two bytes gray-scale\n");
				failed = true;
			}

			if (options.MaxHeight != matrixB16.h || options.MaxWidth != matrixB16.w) {
				fprintf(stderr, "Input and matrix B file parameters mismatch\n");
				failed = true;
			}
		}

		if (failed) {
			fprintf(stderr, "Matrix B file reading error. Ignore parameters\n");
		}
	}

	Sam hSam(options.Info, isTwoBytesMatrixB);
	CHECK_FAST(hSam.Init(
		options,
		options.GrayscaleCorrection.MatrixA != NULL ? matrixA.data.get() : NULL,
		(void *)options.GrayscaleCorrection.MatrixB != NULL ? (isTwoBytesMatrixB ? (void *)matrixB16.data.get() : (void *)matrixB8.data.get()) : NULL
	));
	CHECK_FAST(hSam.Transform(inputImg));

	inputImg.clear();
	CHECK_FAST(hSam.Close());

	return FAST_OK;
}
