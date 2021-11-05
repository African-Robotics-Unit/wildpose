/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "fastvideo_sdk.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "helper_pfm.hpp"
#include "helper_lut.hpp"

#include "Image.h"

#include "DebayerFfmpegMulti.hpp"
#include "CameraMultiSampleOptions.h"

#include "FastAllocator.h"

#include "DecodeError.hpp"

fastStatus_t LoadMatrixA(
	const char *fileName,
	ImageT<float, FastAllocator> &matrix,
	const unsigned maxWidth,
	const unsigned maxHeight
) {
	fastStatus_t ret = FAST_OK;
	if (fileName != NULL) {
		unsigned channels;

		printf("\nMatrix A: %s\n", fileName);
		CHECK_FAST(fvLoadPFM(fileName, matrix.data, matrix.w, matrix.wPitch, FAST_ALIGNMENT, matrix.h, channels));

		if (channels != 1) {
			fprintf(stderr, "Matrix A file must not be color\n");
			ret = FAST_INVALID_FORMAT;
		}

		if (maxHeight != matrix.h || maxWidth != matrix.w) {
			fprintf(stderr, "Input and matrix A file parameters mismatch\n");
			ret = FAST_INVALID_SIZE;
		}

		if (ret != FAST_OK) {
			fprintf(stderr, "Matrix A file reading error. Ignore parameters\n");
		}
	}
	return ret;
}

fastStatus_t LoadMatrixB(
	const char *fileName,
	ImageT<char, FastAllocator> &matrix,
	const unsigned maxWidth,
	const unsigned maxHeight
) {
	fastStatus_t ret = FAST_OK;
	if (fileName != NULL) {
		printf("\nMatrix B: %s\n", fileName);
		ret = fvLoadImage(
			std::string(fileName),
			std::string(""),
			matrix,
			maxHeight,
			maxWidth,
			8, false
		);

		if (matrix.surfaceFmt != FAST_I8) {
			fprintf(stderr, "Matrix B file must not be color\n");
			ret = FAST_INVALID_FORMAT;
		}

		if (maxHeight != matrix.h || maxWidth != matrix.w) {
			fprintf(stderr, "Input and matrix B file parameters mismatch\n");
			ret = FAST_INVALID_SIZE;
		}

		if (ret != FAST_OK) {
			fprintf(stderr, "Matrix B file reading error. Ignore parameters\n");
		}
	}
	return ret;
}

fastStatus_t RunCameraMulti(CameraMultiSampleOptions &options) {
	std::list< Image<FastAllocator> > inputImg;
	CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, options.RawWidth, options.RawHeight, options.BitsPerChannel, false));
	options.SurfaceFmt = FAST_I8;
	for (auto i = inputImg.begin(); i != inputImg.end(); ++i) {
		if ((*i).surfaceFmt != FAST_I8) {
			// validate input images
			fprintf(stderr, "Input file must not be color\n");
			return FAST_IO_ERROR;
		}
	}

	printf("Input surface format: grayscale\n");
	printf("Pattern: %s\n", EnumToString(options.Debayer.BayerFormat));
	printf("Output surface format: %s\n", EnumToString(FAST_RGB8));
	printf("Output sampling format: %s\n", EnumToString(options.JpegEncoder.SamplingFmt));
	printf("Debayer algorithm: %s\n", EnumToString(options.Debayer.BayerType));
	printf("JPEG quality: %d%%\n", options.JpegEncoder.Quality);
	printf("Restart interval: %d\n", options.JpegEncoder.RestartInterval);

	if (options.BaseColorCorrection_0.BaseColorCorrectionEnabled) {
		printf("Correction matrix:\n");
		printf("\t%.3f\t%.3f\t%.3f\t%.3f\n",
			options.BaseColorCorrection_0.BaseColorCorrection[0], options.BaseColorCorrection_0.BaseColorCorrection[1],
			options.BaseColorCorrection_0.BaseColorCorrection[2], options.BaseColorCorrection_0.BaseColorCorrection[3]);
		printf("\t%.3f\t%.3f\t%.3f\t%.3f\n",
			options.BaseColorCorrection_0.BaseColorCorrection[4], options.BaseColorCorrection_0.BaseColorCorrection[5],
			options.BaseColorCorrection_0.BaseColorCorrection[6], options.BaseColorCorrection_0.BaseColorCorrection[7]);
		printf("\t%.3f\t%.3f\t%.3f\t%.3f\n",
			options.BaseColorCorrection_0.BaseColorCorrection[8], options.BaseColorCorrection_0.BaseColorCorrection[9],
			options.BaseColorCorrection_0.BaseColorCorrection[10], options.BaseColorCorrection_0.BaseColorCorrection[11]);
	}
	if (options.BaseColorCorrection_1.BaseColorCorrectionEnabled) {
		printf("Correction matrix:\n");
		printf("\t%.3f\t%.3f\t%.3f\t%.3f\n",
			options.BaseColorCorrection_1.BaseColorCorrection[0], options.BaseColorCorrection_1.BaseColorCorrection[1],
			options.BaseColorCorrection_1.BaseColorCorrection[2], options.BaseColorCorrection_1.BaseColorCorrection[3]);
		printf("\t%.3f\t%.3f\t%.3f\t%.3f\n",
			options.BaseColorCorrection_1.BaseColorCorrection[4], options.BaseColorCorrection_1.BaseColorCorrection[5],
			options.BaseColorCorrection_1.BaseColorCorrection[6], options.BaseColorCorrection_1.BaseColorCorrection[7]);
		printf("\t%.3f\t%.3f\t%.3f\t%.3f\n",
			options.BaseColorCorrection_1.BaseColorCorrection[8], options.BaseColorCorrection_1.BaseColorCorrection[9],
			options.BaseColorCorrection_1.BaseColorCorrection[10], options.BaseColorCorrection_1.BaseColorCorrection[11]);
	}

	ImageT<float, FastAllocator> matrixA_0, matrixA_1;
	CHECK_FAST(LoadMatrixA(options.MAD_0.MatrixA, matrixA_0, options.MaxWidth, options.MaxHeight));
	CHECK_FAST(LoadMatrixA(options.MAD_1.MatrixA, matrixA_1, options.MaxWidth, options.MaxHeight));

	ImageT<char, FastAllocator> matrixB_0, matrixB_1;
	CHECK_FAST(LoadMatrixB(options.MAD_0.MatrixB, matrixB_0, options.MaxWidth, options.MaxHeight));
	CHECK_FAST(LoadMatrixB(options.MAD_1.MatrixB, matrixB_1, options.MaxWidth, options.MaxHeight));

	std::unique_ptr<unsigned char, FastAllocator> lutData_0, lutData_1;
	if (options.Lut != NULL) {
		printf("\nLUT file: %s\n", options.Lut);
		CHECK_FAST(fvLoadLut(options.Lut, lutData_0, 256));
	}
	if (options.Lut_1 != NULL) {
		printf("\nLUT file: %s\n", options.Lut_1);
		CHECK_FAST(fvLoadLut(options.Lut_1, lutData_1, 256));
	}

	DebayerFfmpegMulti hDebayerFfmpeg(false);
	CHECK_FAST(hDebayerFfmpeg.Init(
		options,

		lutData_0,
		options.MAD_0.MatrixA != NULL ? matrixA_0.data.get() : NULL,
		options.MAD_0.MatrixB != NULL ? matrixB_0.data.get() : NULL,

		lutData_1,
		options.MAD_1.MatrixA != NULL ? matrixA_1.data.get() : NULL,
		options.MAD_1.MatrixB != NULL ? matrixB_1.data.get() : NULL
	));
	int cameraId = 0;
	for (auto fileit = inputImg.begin(); fileit != inputImg.end(); fileit++) {
		if (!DecodeError(hDebayerFfmpeg.StoreFrame(*fileit, cameraId))) {
			return FAST_IO_ERROR;
		}
		cameraId = (cameraId + 1) % 2;
	}

	inputImg.clear();
	CHECK_FAST(hDebayerFfmpeg.Close());

	return FAST_OK;
}
