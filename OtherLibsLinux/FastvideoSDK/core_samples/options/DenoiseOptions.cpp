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
#include <string.h>

#include "options/ParametersParser.h"
#include "helper_image/helper_common.h"

#include "DenoiseOptions.h"
#include "StringToEnumDenoise.h"

bool Get123PositiveArgumentsFloat(int argc, const char **argv, const char *string_ref, const char *name, float default_value, float max_value, float *float_retval1, float *float_retval2, float *float_retval3) {
	char *first, *second, *third;
	first = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, argv, string_ref, &first);
	second = first == NULL ? NULL : strchr(first, ';');
	third = second == NULL ? NULL : strchr(second + 1, ';');
	*float_retval1 = first == NULL ? default_value : (float)atof(first);
	if (second == NULL) {
		if (*float_retval1 < 0 || *float_retval1 > max_value) {
			if (ParametersParser::CheckCmdLineFlag(argc, argv, "n"))
				printf("The specified %s value is out of range. Setting it to default.\n", name);
			*float_retval1 = default_value;
		}
		*float_retval2 = *float_retval3 = *float_retval1;
	} else {
		if (*float_retval1 < 0 || *float_retval1 > max_value) {
			printf("The specified %s[0] value is out of range. Setting it to default.\n", name);
			*float_retval1 = default_value;
		}
		*float_retval2 = second == NULL ? *float_retval1 : (float)atof(second + 1);
		if (second != NULL && *float_retval2 < 0 || *float_retval2 > max_value) {
			printf("The specified %s[1] value is out of range. Setting it to default.\n", name);
			*float_retval2 = default_value;
		}
		*float_retval3 = third == NULL ? *float_retval2 : (float)atof(third + 1);
		if (third != NULL && *float_retval3 < 0 || *float_retval3 > max_value) {
			printf("The specified %s[2] value is out of range. Setting it to default.\n", name);
			*float_retval3 = default_value;
		}
	}
	return first != NULL;
}

bool DenoiseOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, (char **)argv)) {
		return false;
	}

	const float default_threshold = 3.0f, max_threshold = 65536.0f;
	const float default_enhance = 1.0f, max_enhance = 256.0f;
	const int default_dwt_levels = 2, max_dwt_levels = 11;

	Threshold[0] = Threshold[1] = Threshold[2] = 0;
	Enhance[0] = Enhance[1] = Enhance[2] = 0;

	if (ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "w", &EnumToString))
		Wavelet = IdentifyWavelet(EnumToString);
	else
		Wavelet = FAST_WAVELET_CDF97;

	if (ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "f", &FunctionName))
		Function = IdentifyThresholdingFunction(FunctionName);
	else
		Function = FAST_THRESHOLD_FUNCTION_SOFT;

	// 1, 2 or 3 values of threshold can be set: single value for all color channels or (Y, Cb/Cr) or (Y, Cb, Cr).
	bool threshold_specified = Get123PositiveArgumentsFloat(argc, (const char **)argv, "t", "threshold", default_threshold, max_threshold, &Threshold[0], &Threshold[1], &Threshold[2]);
	if (!threshold_specified) {
		Threshold[0] = Threshold[1] = Threshold[2] = 1.0f;
	}

	// 1, 2 or 3 values of threshold can be set: single value for all color channels or (Y, Cb/Cr) or (Y, Cb, Cr).
	bool enhance_specified = Get123PositiveArgumentsFloat(argc, (const char **)argv, "e", "enhance", default_enhance, max_enhance, &Enhance[0], &Enhance[1], &Enhance[2]);
	if (!enhance_specified) {
		Enhance[0] = Enhance[1] = Enhance[2] = 1.0f;
	}

	DWT_Levels = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "l") - 1;
	if (!ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "l")) {
		DWT_Levels = default_dwt_levels;
	} else if (DWT_Levels < 0 || DWT_Levels > max_dwt_levels) {
		if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "l"))
			printf("The specified number of resolution levels is out of range. Setting it to default.\n");
		DWT_Levels = default_dwt_levels;
	}

	return true;
}
