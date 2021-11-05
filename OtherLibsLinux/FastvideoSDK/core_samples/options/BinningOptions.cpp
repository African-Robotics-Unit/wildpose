/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "BinningOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <cstring>

fastBinningMode_t BinningOptions::GetBinningModeFromString(const char *pattern) {
	fastBinningMode_t ret = FAST_BINNING_NONE;

	if (pattern != NULL) {
		if (strcmp(pattern, "sum") == 0)
			ret = FAST_BINNING_SUM;
		else if (strcmp(pattern, "avg") == 0)
			ret = FAST_BINNING_AVERAGE;
		else {
			fprintf(stderr, "Pattern %s was not recognized.\nSet to none\n", pattern);
			ret = FAST_BINNING_NONE;
		}
	}

	return ret;
}

bool BinningOptions::Parse(int argc, char *argv[]) {
	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), "mode", &tmp);
	Mode = GetBinningModeFromString(tmp);

	Factor = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "factor");
	if (Factor < 1 || Factor > 4) {
		fprintf(stderr, "Unsupported factor size. Set to default\n");
		Factor = 1;
	}

	return true;
}
