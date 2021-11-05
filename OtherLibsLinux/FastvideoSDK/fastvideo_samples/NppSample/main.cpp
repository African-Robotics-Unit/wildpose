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

#include "RunFilterSamples.hpp"
#include "RunResizeSamples.hpp"
#include "RunRotateSamples.hpp"
#include "NppImageFilterSampleOptions.h"

#include "ParametersParser.h"
#include "FilesParameterChecker.hpp"

#include "Init.hpp"
#include "DecodeError.hpp"

#include "RunPerspective.hpp"
#include "RunRemap.hpp"

int main(int argc, char *argv[]) {
	BaseOptions options;

	bool status = sampleInit(argc, argv, options);
	if (status) {
		if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "gauss")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					NppImageFilterSampleOptions options;

					if (!options.Parse(argc, argv)) {
						fprintf(stderr, "Options parsing error\n");
						status = false;
					}

					if (status) {
						status = DecodeError(RunGauss(options));
					}
				}
				break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "unsharp")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW,
				options.OutputPath, FilesParameterChecker::FAST_RAW
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					NppImageFilterSampleOptions options;

					if (!options.Parse(argc, argv)) {
						fprintf(stderr, "Options parsing error\n");
						status = false;
					}

					if (status) {
						status = DecodeError(RunUnsharpMask(options));
					}
				}
				break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "resize")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					NppResizeSampleOptions options;

					if (!options.Parse(argc, argv)) {
						fprintf(stderr, "Options parsing error\n");
						status = false;
					}

					if (status) {
						status = DecodeError(RunResize(options));
					}
				}
				break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "rotate")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					NppRotateSampleOptions options;

					if (!options.Parse(argc, argv)) {
						fprintf(stderr, "Options parsing error\n");
						status = false;
					}

					if (status) {
						status = DecodeError(RunRotate(options));
					}
				}
				break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "rotate")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					NppRotateSampleOptions options;

					if (!options.Parse(argc, argv)) {
						fprintf(stderr, "Options parsing error\n");
						status = false;
					}

					if (status) {
						status = DecodeError(RunRotate(options));
					}
				}
				break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
			}
		} else if (
			ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "remap") ||
			ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "remap3")
		) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					NppRemapSampleOptions options;

					if (!options.Parse(argc, argv)) {
						fprintf(stderr, "Options parsing error\n");
						status = false;
					}

					if (status) {
						status = DecodeError(RunRemap(options));
					}
				}
				break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "perspective")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW,
				options.OutputPath, FilesParameterChecker::FAST_RAW
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					NppPerspectiveSampleOptions options;

					if (!options.Parse(argc, argv)) {
						fprintf(stderr, "Options parsing error\n");
						status = false;
					}

					if (status) {
						status = DecodeError(RunPerspective(options));
					}
				}
				break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp or ppm.\n");
					status = false;
					break;
			}
		}

		sampleDestroy(options);
	}
	return status ? 0 : -1;
}
