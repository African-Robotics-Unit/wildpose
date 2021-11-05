/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "RawImportOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string.h>

bool RawImportOptions::FormatParser(const char *rawFormat) {
	if (strcmp(rawFormat, "ximea12") == 0) {
		RawFormat = FAST_RAW_XIMEA12;
	} else if (strcmp(rawFormat, "ptg12") == 0) {
		RawFormat = FAST_RAW_PTG12;
	} else {
		fprintf(stderr, "Incorrect RAW format (%s)\n", rawFormat);
		return false;
	}

	return true;
}

bool RawImportOptions::Parse(int argc, char *argv[]) {
	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), "format", &tmp);
	if (tmp != NULL) {
		if (!FormatParser(tmp)) {
			return false;
		}
	} else {
		fprintf(stderr, "-format parameter was not found\n");
		return false;
	}

	Width = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "width");
	if (Width <= 0) {
		fprintf(stderr, "-width parameter was not found\n");
		return false;
	}

	Height = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "height");
	if (Height <= 0) {
		fprintf(stderr, "-height parameter was not found\n");
		return false;
	}

	IsGpu = ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "gpu");

	IsConvert12to16 = ParametersParser::CheckCmdLineFlag(argc, const_cast<const char**>(argv), "convert16");


	return true;
}
