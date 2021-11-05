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

#include "RunAffine.hpp"
#include "RunMedian.hpp"
#include "RunSam.hpp"
#include "RunBGRXImport.hpp"
#include "RunLuts.hpp"
#include "RunBaseColorCorrection.hpp"
#include "RunToneCurve16.hpp"
#include "RunBitDepthConverter.hpp"
#include "RunBayerBlackShift.hpp"
#include "RunSelectChannel.hpp"
#include "RunColorConvertion.hpp"
#include "RunDefringe.hpp"
#include "RunBinning.hpp"
#include "RunBadPixelCorrection.hpp"
#include "RunCrop.hpp"

#include "ParametersParser.h"
#include "FilesParameterChecker.hpp"

#include "Help.h"
#include "Init.hpp"
#include "DecodeError.hpp"
#include "RunSharpen.hpp"

int main(int argc, char *argv[]) {
	BaseOptions options;

	bool status = sampleInit(argc, argv, options);
	if (status) {
		if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "affine")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW,
				options.OutputPath, FilesParameterChecker::FAST_RAW
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					AffineSampleOptions affineOptions;
					if (!affineOptions.Parse(argc, argv)) {
						helpPrint();
						status = false;
					}
					if (status) {
						status = DecodeError(RunAffine(affineOptions));
					}
				}
				break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "toneCurve")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					ToneCurveSampleOptions options;
					if (!options.Parse(argc, argv)) {
						helpPrint();
						status = false;
					}
					if (status) {
						status = DecodeError(RunToneCurve(options));
					}
				}
				break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
					status = false;
					break;
			}
		}
		else if (ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "sharp")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
			case FilesParameterChecker::FAST_OK:
			{
				ImageFilterSampleOptions options;
				if (!options.Parse(argc, argv)) {
					helpPrint();
					status = false;
				}
				if (status) {
					status = DecodeError(RunSharpen(options, true));
				}
			}
			break;
			case FilesParameterChecker::FAST_INPUT_ERROR:
				fprintf(stderr, "Input file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
				status = false;
				break;
			case FilesParameterChecker::FAST_OUTPUT_ERROR:
				fprintf(stderr, "Output file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
				status = false;
				break;
			case FilesParameterChecker::FAST_BOTH_ERROR:
				fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
				status = false;
				break;
			}
		}
		else if (ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "blur")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
			case FilesParameterChecker::FAST_OK:
			{
				ImageFilterSampleOptions options;
				if (!options.Parse(argc, argv)) {
					helpPrint();
					status = false;
				}
				if (status) {
					status = DecodeError(RunSharpen(options, false));
				}
			}
			break;
			case FilesParameterChecker::FAST_INPUT_ERROR:
				fprintf(stderr, "Input file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
				status = false;
				break;
			case FilesParameterChecker::FAST_OUTPUT_ERROR:
				fprintf(stderr, "Output file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
				status = false;
				break;
			case FilesParameterChecker::FAST_BOTH_ERROR:
				fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp, ppm or pgm.\n");
				status = false;
				break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "bgrxImport")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					status = DecodeError(RunBGRXImport(options));
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
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "sam16")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_GRAY,
				options.OutputPath, FilesParameterChecker::FAST_RAW_GRAY
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					DebayerSampleOptions madOptions;
					if (!madOptions.Parse(argc, argv)) {
						helpPrint();
						status = false;
					}

					if (status) {
						status = DecodeError(RunSam(madOptions, true));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "crop")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW,
				options.OutputPath, FilesParameterChecker::FAST_RAW
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					CropSampleOptions cropOptions;
					if (!cropOptions.Parse(argc, argv)) {
						helpPrint();
						status = false;
					}

					if (status) {
						status = DecodeError(RunCrop(cropOptions));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be pgm or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be pgm or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be pgm or ppm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "binning")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_GRAY,
				options.OutputPath, FilesParameterChecker::FAST_RAW_GRAY
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					BinningSampleOptions binningOptions;
					if (!binningOptions.Parse(argc, argv)) {
						helpPrint();
						status = false;
					}

					if (status) {
						status = DecodeError(RunBinning(binningOptions));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "sam")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_GRAY,
				options.OutputPath, FilesParameterChecker::FAST_RAW_GRAY
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					DebayerSampleOptions madOptions;
					if (!madOptions.Parse(argc, argv)) {
						helpPrint();
						status = false;
					}

					if (status) {
						status = DecodeError(RunSam(madOptions, false));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "bayerBlackShift")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_GRAY,
				options.OutputPath, FilesParameterChecker::FAST_RAW_GRAY
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					BayerBlackShiftSampleOptions blackShiftOptions;
					if (!blackShiftOptions.Parse(argc, argv)) {
						helpPrint();
						status = false;
					}

					if (status) {
						status = DecodeError(RunBayerBlackShift(blackShiftOptions));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "badPixelCorrection")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_GRAY,
				options.OutputPath, FilesParameterChecker::FAST_RAW_GRAY
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					BadPixelCorrectionSampleOptions options;

					if (!options.Parse(argc, argv)) {
						fprintf(stderr, "Options parsing error\n");
						status = false;
					}

					if (status) {
						status = DecodeError(RunBadPixelCorrection(options));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be pgm format.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be pgm format.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "median")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW,
				options.OutputPath, FilesParameterChecker::FAST_RAW
			)) {
				case FilesParameterChecker::FAST_OK:
					if (status) {
						status = DecodeError(RunMedianFilter(options));
					}
					break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be pgm or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be pgm or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be pgm or ppm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "bitDepthConverter")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW,
				options.OutputPath, FilesParameterChecker::FAST_RAW
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					BitDepthConverterSampleOptions bitDepthOptions;
					if (!bitDepthOptions.Parse(argc, argv)) {
						helpPrint();
						status = false;
					}

					if (status) {
						status = DecodeError(RunBitDepthConverter(bitDepthOptions));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be pgm or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be pgm or ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be pgm or ppm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "selectChannel")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_GRAY
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					SelectChannelSampleOptions selectOptions;
					if (!selectOptions.Parse(argc, argv)) {
						helpPrint();
						status = false;
					}

					if (status) {
						status = DecodeError(RunSelectChannel(selectOptions));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be pgm or ppm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "defringe")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					DefringeSampleOptions defringeOptions;
					if (!defringeOptions.Parse(argc, argv)) {
						helpPrint();
						status = false;
					}

					if (status) {
						status = DecodeError(RunDefringe(defringeOptions));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be ppm.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "colorConvertion")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_GRAY
			)) {
				case FilesParameterChecker::FAST_OK:
					if (status) {
						status = DecodeError(RunRgbToGrayscale(options));
					}
					break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
				{
					switch (FilesParameterChecker::Validate(
						options.InputPath, FilesParameterChecker::FAST_RAW_GRAY,
						options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
					)) {
						case FilesParameterChecker::FAST_OK:
							if (status) {
								status = DecodeError(RunGrayscaleToRgb(options));
							}
							break;
						case FilesParameterChecker::FAST_INPUT_ERROR:
							fprintf(stderr, "Input file has inappropriate format.\nShould be pgm.\n");
							status = false;
							break;
						case FilesParameterChecker::FAST_OUTPUT_ERROR:
							fprintf(stderr, "Output file has inappropriate format.\nShould be ppm.\n");
							status = false;
							break;
						case FilesParameterChecker::FAST_BOTH_ERROR:
							fprintf(stderr, "Input and output file has inappropriate format.\nShould be pgm or ppm.\n");
							status = false;
							break;
					}
				}
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut8_16b") ||
			ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut10_16b") ||
			ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_16b") ||
			ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut14_16b") ||
			ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_16b")
		) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_GRAY,
				options.OutputPath, FilesParameterChecker::FAST_RAW_GRAY
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					LutDebayerSampleOptions lutOptions;

					if (!lutOptions.Parse(argc, argv)) {
						fprintf(stderr, "Options parsing error\n");
						status = false;
					}

					if (status) {
						status = DecodeError(RunLutBayer(lutOptions));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be pgm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be pgm format.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be pgm format.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "hsvLut3D")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					LutSampleOptions options;
					if (!options.Parse(argc, argv)) {
						fprintf(stderr, "Options parsing error\n");
						status = false;
					}
					if (status) {
						status = DecodeError(RunLutHsv3D(options));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be ppm format.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be ppm format.\n");
					status = false;
					break;
			}
		} else if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "rgbLut3D")) {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
				case FilesParameterChecker::FAST_OK:
				{
					LutSampleOptions options;
					if (!options.Parse(argc, argv)) {
						fprintf(stderr, "Options parsing error\n");
						status = false;
					}
					if (status) {
						status = DecodeError(RunLutRgb3D(options));
					}
					break;
				}
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be ppm.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be ppm format.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be ppm format.\n");
					status = false;
					break;
			}
		} else {
			switch (FilesParameterChecker::Validate(
				options.InputPath, FilesParameterChecker::FAST_RAW_COLOR,
				options.OutputPath, FilesParameterChecker::FAST_RAW_COLOR
			)) {
				case FilesParameterChecker::FAST_OK:
					if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut8")) {
						LutSampleOptions lutOptions;
						if (!lutOptions.Parse(argc, argv)) {
							helpPrint();
							status = false;
						}
						if (status) {
							status = DecodeError(RunLut8(lutOptions));
						}
					} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut8c")) {
						LutSampleOptions lutOptions;
						if (!lutOptions.Parse(argc, argv)) {
							helpPrint();
							status = false;
						}
						if (status) {
							status = DecodeError(RunLut8c(lutOptions));
						}
					} else if (
						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_12") ||
						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_12c") ||

						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_8") ||
						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_8c") ||

						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_16") ||
						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_16c") ||

						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_16") ||
						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_16c") ||

						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_8") ||
						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_8c") ||

						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_16_fr") ||
						ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_16c_fr")
					) {
						LutSampleOptions lutOptions;
						if (!lutOptions.Parse(argc, argv)) {
							helpPrint();
							status = false;
						}
						if (status) {
							status = DecodeError(RunLut16(lutOptions));
						}
					} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "baseColorCorrection")) {
						BaseColorCorrectionSampleOptions baseColorCorrectionOptions;
						if (!baseColorCorrectionOptions.Parse(argc, argv)) {
							helpPrint();
							status = false;
						}
						if (status) {
							status = DecodeError(RunBaseColorCorrection(baseColorCorrectionOptions));
						}
					} else {
						helpPrint();
					}
					break;
				case FilesParameterChecker::FAST_INPUT_ERROR:
					fprintf(stderr, "Input file has inappropriate format.\nShould be bmp.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_OUTPUT_ERROR:
					fprintf(stderr, "Output file has inappropriate format.\nShould be bmp format.\n");
					status = false;
					break;
				case FilesParameterChecker::FAST_BOTH_ERROR:
					fprintf(stderr, "Input and output file has inappropriate format.\nShould be bmp format.\n");
					status = false;
					break;
			}
		}

		sampleDestroy(options);
	}
	return status ? 0 : -1;
}
