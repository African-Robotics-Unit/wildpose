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

#include "RunJpegAsync.hpp"

#include "JpegEncoderSampleOptions.h"
#include "FilesParameterChecker.hpp"

#include "Init.hpp"
#include "DecodeError.hpp"

int main(int argc, char *argv[]) {
	JpegEncoderSampleOptions options;

	bool status = sampleInit(argc, argv, options);
	if (status) {
		switch (FilesParameterChecker::Validate(
			options.InputPath, FilesParameterChecker::FAST_RAW,
			options.OutputPath, FilesParameterChecker::FAST_JPEG
		)) {
			case FilesParameterChecker::FAST_OK:
				status = DecodeError(RunJpegAsync(options));
				break;
			case FilesParameterChecker::FAST_INPUT_ERROR:
				fprintf(stderr, "Input file has inappropriate format.\nFor encoding input file should be in bmp, pgm, ppm.\n");
				status = false;
				break;
			case FilesParameterChecker::FAST_OUTPUT_ERROR:
				fprintf(stderr, "Output file has inappropriate format.\nFor encoding output file should be in jpg\n");
				status = false;
				break;
			case FilesParameterChecker::FAST_BOTH_ERROR:
				fprintf(stderr, "Input and output files has inappropriate format.\nInput file should be in bmp, pgm or ppm. Output file should be in jpg\n");
				status = false;
				break;
		}

		sampleDestroy(options);
	}
	return status ? 0 : -1;
}
