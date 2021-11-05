/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "RunImageConverter.hpp"
#include "RunMosaic.hpp"
#include "RunMatrixGenerator.hpp"
#include "RunCrop.hpp"
#include "RunColorConvert.hpp"

#include "BaseOptions.h"
#include "ImageConverterSampleOptions.h"
#include "MatrixGeneratorSampleOptions.h"

#include "Help.h"
#include "DecodeError.hpp"
#include "ParametersParser.h"

int main(int argc, char *argv[]) {
	BaseOptions baseOptions;

	if (!baseOptions.Parse(argc, argv)) {
		helpPrint();
		return -1;
	}

	if (baseOptions.Help) {
		helpPrint();
		return 0;
	}

	fastStatus_t ret = FAST_OK;
	if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "pattern")) {
		ret = RunMosaic(argc, argv);
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "matrix")) {
		MatrixGeneratorSampleOptions options;
		if (!options.Parse(argc, argv)) {
			helpPrint();
			return -1;
		}
		ret = RunMatrixGenerator(options);
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "grayscale")) {
		BaseOptions options;
		if (!options.Parse(argc, argv)) {
			helpPrint();
			return -1;
		}
		ret = RunColorConvert(options);
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "crop")) {
		CropSampleOptions options;
		if (!options.Parse(argc, argv)) {
			helpPrint();
			return -1;
		}
		ret = RunCrop(options);
	} else {
		ImageConverterSampleOptions options;
		if (!options.Parse(argc, argv)) {
			helpPrint();
			return -1;
		}
		ret = RunConversion(options);
	}
	return DecodeError(ret) ? 0 : -1;
}
