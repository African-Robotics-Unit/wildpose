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
#include <string>
#include <string.h>
#include <thread>
#include <math.h> // for gcc build

#include "ParametersParser.h"
#include "helper_common.h"
#include "helper_dir.hpp"

#include "J2kEncoderOptions.h"


fastEncoderJ2kAlgorithmType_t IdentifyAlgorithm(char *algorithmName) {
	fastEncoderJ2kAlgorithmType_t name = FAST_ENCODER_J2K_ALGORITHM_UNKNOWN;

	if (strcmp(algorithmName, "rev") == 0) name = FAST_ENCODER_J2K_ALGORITHM_ENCODE_REVERSIBLE;
	if (strcmp(algorithmName, "irrev") == 0) name = FAST_ENCODER_J2K_ALGORITHM_ENCODE_IRREVERSIBLE;

	return name;
}

bool J2kEncoderOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	const float default_quality = 100, max_quality = 150;
	const int default_dwt_levels = 7, max_dwt_levels = 11;

	if (ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "a", &AlgorithmName))
		Algorithm = IdentifyAlgorithm(AlgorithmName);
	else
		Algorithm = FAST_ENCODER_J2K_ALGORITHM_ENCODE_IRREVERSIBLE;

	if (InputPath != NULL) InputFilesize = fileSize(InputPath);
	if (Algorithm == FAST_ENCODER_J2K_ALGORITHM_ENCODE_IRREVERSIBLE) {
		Quality = (float)ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "q");
		if (Quality <= 0 || Quality > max_quality) {
			Quality = default_quality;
		}
		Quality *= 0.01f;
	} else {
		Quality = 1.0f;
	}

	TargetFilesize = 0;
	CompressionRatio = (float)ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "cr");
	if (CompressionRatio > 1) {
		TargetFilesize = (long)floor(InputFilesize / CompressionRatio);
	} else {
		if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "cr"))
			printf("The specified compression ratio value is out of range.\n");
		CompressionRatio = 0;
	}

	DWT_Levels = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "l", default_dwt_levels) - 1;
	if (DWT_Levels < 0 || DWT_Levels > max_dwt_levels) {
		if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "l"))
			printf("The specified number of resolution levels is out of range. Setting it to default.\n");
		DWT_Levels = default_dwt_levels;
	}

	BatchSize = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "b", 1);
	if (BatchSize < 1) {
		if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "b"))
			printf("The specified batch size is invalid. Setting it to default.\n");
		BatchSize = 1;
	}

	CodeblockSize = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "c");
	if (CodeblockSize != 8 &&
		CodeblockSize != 16 &&
		CodeblockSize != 32 &&
		CodeblockSize != 64) {
		if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "c"))
			printf("The specified codeblock size is invalid. Setting it to default.\n");
		CodeblockSize = 0; // the default value is defined in "Encoder.cpp"
	}

	Tier2Threads = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "t");
	if (Tier2Threads <= 0) {
		int nthreads = std::thread::hardware_concurrency(); // CPU cores * hyperthreading
		if (nthreads < 4) Tier2Threads = nthreads;
		if (nthreads == 4) Tier2Threads = 3;
		if (nthreads > 4) Tier2Threads = 4;
	}

	Discard = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "discard");

	NoHeader = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "noHeader");
    NoMCT = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "nomct");

	TileWidth = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "tileWidth");
	if (TileWidth < 0) TileWidth = 0;
	TileHeight = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "tileHeight");
	if (TileHeight < 0) TileHeight = 0;

	ss1_x = ss1_y = ss2_x = ss2_y = ss3_x = ss3_y = 1;

	int subsampling = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)(const char **)argv, "s");

	OutputBitDepth = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)(const char **)argv, "outputBitdepth", 0);

	OverwriteSurfaceBitDepth = ParametersParser::GetCmdLineArgumentInt(argc, (const char**)(const char**)argv, "overwriteSourceBitdepth", 0);

    Timeout = ParametersParser::GetCmdLineArgumentFloat(argc, (const char**)argv, "timeout");
    if (Timeout < 0)
    {
        if (ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "timeout"))
            printf("The specified timeout value is invalid. Setting it to default.\n");
        Timeout = 0;
    }

	switch (subsampling)
	{

	case 422:
		ss1_x = ss1_y = 1;
		ss2_x = 2;
		ss2_y = 1;
		ss3_x = 2;
		ss3_y = 1;
		break;
	case 420:
		ss1_x = ss1_y = 1;
		ss2_x = ss2_y = ss3_x = ss3_y = 2;
		break;
	}

	return true;
}
