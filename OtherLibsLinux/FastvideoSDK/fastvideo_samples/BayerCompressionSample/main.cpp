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
#include "RunBayerCompression.hpp"

#include "DebayerJpegSampleOptions.h"
#include "FilesParameterChecker.hpp"

#include "Init.hpp"
#include "DecodeError.hpp"

int main(int argc, char *argv[]) {
	DebayerJpegSampleOptions options;

	bool status = sampleInit(argc, argv, options);
	if (status) {
		switch (FilesParameterChecker::Validate(
			options.InputPath, FilesParameterChecker::FAST_RAW_GRAY,
			options.OutputPath, FilesParameterChecker::FAST_JPEG
		)) {
			case FilesParameterChecker::FAST_OK:
				status = DecodeError(RunSplitterEncoder(options));
				break;
			case FilesParameterChecker::FAST_INPUT_ERROR:
				fprintf(stderr, "Input file has inappropriate format.\nShould be pgm.\n");
				status = false;
				break;
			case FilesParameterChecker::FAST_OUTPUT_ERROR:
				fprintf(stderr, "Output file has inappropriate format.\nShould be jpg or jpeg.\n");
				status = false;
				break;
			case FilesParameterChecker::FAST_BOTH_ERROR:
			{
				switch (FilesParameterChecker::Validate(
					options.InputPath, FilesParameterChecker::FAST_JPEG,
					options.OutputPath, FilesParameterChecker::FAST_RAW_GRAY
				)) {
					case FilesParameterChecker::FAST_OK:
						status = DecodeError(RunDecoderMerger(options));
						break;
					case FilesParameterChecker::FAST_INPUT_ERROR:
						fprintf(stderr, "Input file has inappropriate format.\nShould be jpg or jpeg.\n");
						status = false;
						break;
					case FilesParameterChecker::FAST_OUTPUT_ERROR:
						fprintf(stderr, "Output file has inappropriate format.\nShould be pgm.\n");
						status = false;
						break;
					case FilesParameterChecker::FAST_BOTH_ERROR:
						fprintf(stderr, "Input and output file has inappropriate format.\nInput file should be jpg or jpeg. Output file should be pgm.\n");
						status = false;
						break;
				}
			}
		}

		sampleDestroy(options);
	}
	return status ? 0 : -1;
}
