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

#include "FastAllocator.h"
#include "checks.h"
#include "Image.h"

#include "SurfaceTraits.hpp"
#include "supported_files.hpp"

#include "MatrixGeneratorSampleOptions.h"

#include "helper_pfm.hpp"
#include "ppm.h"

fastStatus_t RunMatrixGenerator(MatrixGeneratorSampleOptions options) {
	Image<FastAllocator> inputImg;

	CHECK_FAST(fvLoadImage(options.InputPath, "", inputImg, 0, 0, 0, false));

	if (BaseOptions::CheckFileExtension(options.OutputPath, ".pfm")) {
		printf("Pixel value: %.3f\n", options.Matrix.PixelValue);

		// PFM-file
		float *data = new float[inputImg.w * inputImg.h];
		for (unsigned i = 0; i < inputImg.w * inputImg.h; i++) {
			data[i] = options.Matrix.PixelValue;
		}

		CHECK_FAST(fvSavePFM(
			options.OutputPath,
			data,
			inputImg.w, inputImg.w,
			inputImg.h, 1
		));

		delete[] data;
	} else {
		printf("Pixel value: %d\n", (int)options.Matrix.PixelValue);

		// PGM-file
		const unsigned pitch = inputImg.w * uDivUp(inputImg.bitsPerChannel, 8u);
		unsigned char *data = new unsigned char[pitch * inputImg.h];
		if (inputImg.bitsPerChannel == 8) {
			for (unsigned i = 0; i < inputImg.w * inputImg.h; i++) {
				data[i] = (int)options.Matrix.PixelValue;
			}
		} else {
			unsigned short *tmp = (unsigned short *)data;
			for (unsigned i = 0; i < inputImg.w * inputImg.h; i++) {
				tmp[i] = (int)options.Matrix.PixelValue;
			}
		}

		SavePPM(
			options.OutputPath,
			data,
			inputImg.w, pitch,
			inputImg.h, inputImg.bitsPerChannel, 1
		);

		delete[] data;
	}

	return FAST_OK;
}