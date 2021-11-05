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
#include <cstring>

#include "DebayerOptions.h"
#include "ParametersParser.h"

fastBayerPattern_t DebayerOptions::GetBayerPatternFromString(const char *pattern) {
	fastBayerPattern_t ret = FAST_BAYER_RGGB;

	if (pattern != NULL) {
		if (strcmp(pattern, "GRBG") == 0)
			ret = FAST_BAYER_GRBG;
		else if (strcmp(pattern, "BGGR") == 0)
			ret = FAST_BAYER_BGGR;
		else if (strcmp(pattern, "GBRG") == 0)
			ret = FAST_BAYER_GBRG;
		else if (strcmp(pattern, "RGGB") == 0)
			ret = FAST_BAYER_RGGB;
		else if (strcmp(pattern, "none") == 0)
			ret = FAST_BAYER_NONE;
		else {
			fprintf(stderr, "Pattern %s was not recognized.\nSet to RGGB\n", pattern);
			ret = FAST_BAYER_RGGB;
		}
	}

	return ret;
}

fastDebayerType_t DebayerOptions::GetBayerAlgorithmType(const char *pattern) {
	fastDebayerType_t ret = FAST_DFPD;

	if (pattern != NULL) {
		if (strcmp(pattern, "DFPD") == 0)
			ret = FAST_DFPD;
		else if (strcmp(pattern, "HQLI") == 0)
			ret = FAST_HQLI;
		else if (strcmp(pattern, "L7") == 0)
			ret = FAST_L7;
		else if (strcmp(pattern, "MG") == 0)
			ret = FAST_MG;
		else if (strcmp(pattern, "MG_V2") == 0)
			ret = FAST_MG_V2;
		else if (strcmp(pattern, "binning_2x2") == 0)
			ret = FAST_BINNING_2x2;
		else if (strcmp(pattern, "binning_4x4") == 0)
			ret = FAST_BINNING_4x4;
		else if (strcmp(pattern, "binning_8x8") == 0)
			ret = FAST_BINNING_8x8;
		else if (strcmp(pattern, "AMAZE") == 0)
			ret = FAST_AMAZE;
		else {
			fprintf(stderr, "Pattern %s was not recognized.\nSet to DFPD\n", pattern);
			ret = FAST_DFPD;
		}
	}

	return ret;
}

bool DebayerOptions::Parse(int argc, char *argv[]) {
	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), "type", &tmp);
	BayerType = GetBayerAlgorithmType(tmp);

	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "pattern", &tmp);
	BayerFormat = GetBayerPatternFromString(tmp);

	return true;
}
