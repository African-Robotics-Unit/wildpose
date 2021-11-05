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

#include "RunFfmpeg.hpp"

#include "FfmpegSampleOptions.h"
#include "FilesParameterChecker.hpp"

#include "Help.h"
#include "Init.hpp"
#include "DecodeError.hpp"

int main(int argc, char *argv[]) {
	BaseOptions options;

	bool status = sampleInit(argc, argv, options);
	if (status) {
		switch (FilesParameterChecker::Validate(
			options.InputPath, FilesParameterChecker::FAST_RAW,
			options.OutputPath, FilesParameterChecker::FAST_AVI
		)) {
			case FilesParameterChecker::FAST_OK:
			{
				FfmpegSampleOptions encoderOptions;
				if (!encoderOptions.Parse(argc, argv)) {
					helpPrint();
					return -1;
				}

				status = DecodeError(RunFfmpegEncode(encoderOptions));
				break;
			}
			case FilesParameterChecker::FAST_INPUT_ERROR:
				fprintf(stderr, "Input file has inappropriate format.\nShould be ppm, bmp for encoding and avi for decoding.\n");
				status = false;
				break;
			case FilesParameterChecker::FAST_OUTPUT_ERROR:
				fprintf(stderr, "Input file has inappropriate format.\nShould be ppm, bmp for decoding and avi for encoding.\n");
				status = false;
				break;
			case FilesParameterChecker::FAST_BOTH_ERROR:
			{
				switch (FilesParameterChecker::Validate(
					options.InputPath, FilesParameterChecker::FAST_AVI,
					options.OutputPath, FilesParameterChecker::FAST_RAW
				)) {
					case FilesParameterChecker::FAST_OK:
						options.SurfaceFmt = BaseOptions::GetSurfaceFormatFromExtension(options.OutputPath);
						status = DecodeError(RunFfmpegDecode(options));
						break;
					case FilesParameterChecker::FAST_INPUT_ERROR:
						fprintf(stderr, "Input file has inappropriate format.\nShould be ppm, bmp for encoding and avi for decoding.\n");
						status = false;
						break;
					case FilesParameterChecker::FAST_OUTPUT_ERROR:
						fprintf(stderr, "Input file has inappropriate format.\nShould be ppm, bmp for decoding and avi for encoding.\n");
						status = false;
						break;
					case FilesParameterChecker::FAST_BOTH_ERROR:
						fprintf(stderr, "Input and output file has inappropriate format.\nInput and output files should be ppm, bmp or avi\n");
						status = false;
						break;
				}
				break;
			}
		}

		sampleDestroy(options);
	}
	return status ? 0 : -1;
}
