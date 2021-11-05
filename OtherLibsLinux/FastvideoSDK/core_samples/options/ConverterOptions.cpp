/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "ConverterOptions.h"
#include "ParametersParser.h"

#include <cstring>

fastRawFormat_t ConverterOptions::GetRawFormat(const char* str) {
	if (strcmp(str, "ptg12") == 0) {
		return FAST_RAW_PTG12;
	}
	if (strcmp(str, "ximea12") == 0) {
		return FAST_RAW_XIMEA12;
	}

	return FAST_RAW_XIMEA12;
}


bool ConverterOptions::Parse(int argc, char *argv[]) {
	Shift = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "shift");
	Randomize = ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "randomize");

	RawFormat = FAST_RAW_XIMEA12;
	char *tmp = NULL;
	if (ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), "format", &tmp)) {
		RawFormat = GetRawFormat(tmp);
	}

	return true;
}
