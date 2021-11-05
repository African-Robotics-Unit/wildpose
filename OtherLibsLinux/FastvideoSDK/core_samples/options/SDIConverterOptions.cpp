/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "SDIConverterOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <cstring>

bool SDIConverterOptions::FormatParser(const char *sdiFormat) {
	if (strcmp(sdiFormat, "YV12_601_FR") == 0) {
		SDIFormat = FAST_SDI_YV12_BT601_FR;
	} else if (strcmp(sdiFormat, "YV12_601") == 0) {
		SDIFormat = FAST_SDI_YV12_BT601;
	} else if (strcmp(sdiFormat, "YV12_709") == 0) {
		SDIFormat = FAST_SDI_YV12_BT709;
	} else if (strcmp(sdiFormat, "YV12_2020") == 0) {
		SDIFormat = FAST_SDI_YV12_BT2020;
	} else if (strcmp(sdiFormat, "YCbCr420_601_FR") == 0) {
		SDIFormat = FAST_SDI_420_8_YCbCr_PLANAR_BT601_FR;
	} else if (strcmp(sdiFormat, "YCbCr420_601") == 0) {
		SDIFormat = FAST_SDI_420_8_YCbCr_PLANAR_BT601;
	} else if (strcmp(sdiFormat, "YCbCr420_709") == 0) {
		SDIFormat = FAST_SDI_420_8_YCbCr_PLANAR_BT709;
	} else if (strcmp(sdiFormat, "YCbCr420_2020") == 0) {
		SDIFormat = FAST_SDI_420_8_YCbCr_PLANAR_BT2020;
	} else if (strcmp(sdiFormat, "NV12_601_FR") == 0) {
		SDIFormat = FAST_SDI_NV12_BT601_FR;
	} else if (strcmp(sdiFormat, "NV12_601") == 0) {
		SDIFormat = FAST_SDI_NV12_BT601;
	} else if (strcmp(sdiFormat, "NV12_709") == 0) {
		SDIFormat = FAST_SDI_NV12_BT709;
	} else if (strcmp(sdiFormat, "NV12_2020") == 0) {
		SDIFormat = FAST_SDI_NV12_BT2020;
	} else if (strcmp(sdiFormat, "P010_601_FR") == 0) {
		SDIFormat = FAST_SDI_P010_BT601_FR;
	} else if (strcmp(sdiFormat, "P010_601") == 0) {
		SDIFormat = FAST_SDI_P010_BT601;
	} else if (strcmp(sdiFormat, "P010_709") == 0) {
		SDIFormat = FAST_SDI_P010_BT709;
	} else if (strcmp(sdiFormat, "P010_2020") == 0) {
		SDIFormat = FAST_SDI_P010_BT2020;
	} else if (strcmp(sdiFormat, "YCbCr420_10_601_FR") == 0) {
		SDIFormat = FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601_FR;
	} else if (strcmp(sdiFormat, "YCbCr420_10_601") == 0) {
		SDIFormat = FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601;
	} else if (strcmp(sdiFormat, "YCbCr420_10_709") == 0) {
		SDIFormat = FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT709;
	} else if (strcmp(sdiFormat, "YCbCr420_10_2020") == 0) {
		SDIFormat = FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT2020;
	} else if (strcmp(sdiFormat, "YCbCr422_601_FR") == 0) {
		SDIFormat = FAST_SDI_422_8_YCbCr_PLANAR_BT601_FR;
	} else if (strcmp(sdiFormat, "YCbCr422_601") == 0) {
		SDIFormat = FAST_SDI_422_8_YCbCr_PLANAR_BT601;
	} else if (strcmp(sdiFormat, "YCbCr422_709") == 0) {
		SDIFormat = FAST_SDI_422_8_YCbCr_PLANAR_BT709;
	} else if (strcmp(sdiFormat, "YCbCr422_2020") == 0) {
		SDIFormat = FAST_SDI_422_8_YCbCr_PLANAR_BT2020;
	} else if (strcmp(sdiFormat, "YCbCr422_10_601_FR") == 0) {
		SDIFormat = FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601_FR;
	} else if (strcmp(sdiFormat, "YCbCr422_10_601") == 0) {
		SDIFormat = FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601;
	} else if (strcmp(sdiFormat, "YCbCr422_10_709") == 0) {
		SDIFormat = FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT709;
	} else if (strcmp(sdiFormat, "YCbCr422_10_2020") == 0) {
		SDIFormat = FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT2020;
	} else if (strcmp(sdiFormat, "CbYCr422_709") == 0) {
		SDIFormat = FAST_SDI_422_8_CbYCrY_BT709;
	} else if (strcmp(sdiFormat, "CbYCr422_2020") == 0) {
		SDIFormat = FAST_SDI_422_8_CbYCrY_BT2020;
	} else if (strcmp(sdiFormat, "CbYCr422_601_FR") == 0) {
		SDIFormat = FAST_SDI_422_8_CbYCrY_BT601_FR;
	} else if (strcmp(sdiFormat, "CbYCr422_601") == 0) {
		SDIFormat = FAST_SDI_422_8_CbYCrY_BT601;
	} else if (strcmp(sdiFormat, "CbYCr422_10_709") == 0) {
		SDIFormat = FAST_SDI_422_10_CbYCrY_PACKED_BT709;
	} else if (strcmp(sdiFormat, "CbYCr422_10_2020") == 0) {
		SDIFormat = FAST_SDI_422_10_CbYCrY_PACKED_BT2020;
	} else if (strcmp(sdiFormat, "CbYCr422_10_601_FR") == 0) {
		SDIFormat = FAST_SDI_422_10_CbYCrY_PACKED_BT601_FR;
	} else if (strcmp(sdiFormat, "CbYCr422_10_601") == 0) {
		SDIFormat = FAST_SDI_422_10_CbYCrY_PACKED_BT601;
	} else if (strcmp(sdiFormat, "CrYCb422_709") == 0) {
		SDIFormat = FAST_SDI_422_8_CrYCbY_BT709;
	} else if (strcmp(sdiFormat, "CrYCb422_2020") == 0) {
		SDIFormat = FAST_SDI_422_8_CrYCbY_BT2020;
	} else if (strcmp(sdiFormat, "CrYCb422_601") == 0) {
		SDIFormat = FAST_SDI_422_8_CrYCbY_BT601;
	} else if (strcmp(sdiFormat, "CrYCb422_601_FR") == 0) {
		SDIFormat = FAST_SDI_422_8_CrYCbY_BT601_FR;
	} else if (strcmp(sdiFormat, "YCbCr444_709") == 0) {
		SDIFormat = FAST_SDI_444_8_YCbCr_PLANAR_BT709;
	} else if (strcmp(sdiFormat, "YCbCr444_2020") == 0) {
		SDIFormat = FAST_SDI_444_8_YCbCr_PLANAR_BT2020;
	}	else if (strcmp(sdiFormat, "YCbCr444_601") == 0) {
		SDIFormat = FAST_SDI_444_8_YCbCr_PLANAR_BT601;
	} else if (strcmp(sdiFormat, "YCbCr444_601_FR") == 0) {
		SDIFormat = FAST_SDI_444_8_YCbCr_PLANAR_BT601_FR;
	} else if (strcmp(sdiFormat, "YCbCr444_10_709") == 0) {
		SDIFormat = FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT709;
	} else if (strcmp(sdiFormat, "YCbCr444_10_2020") == 0) {
		SDIFormat = FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT2020;
	} else if (strcmp(sdiFormat, "YCbCr444_10_601") == 0) {
		SDIFormat = FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601;
	} else if (strcmp(sdiFormat, "YCbCr444_10_601_FR") == 0) {
		SDIFormat = FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601_FR;
	} else if (strcmp(sdiFormat, "RGBA") == 0) {
		SDIFormat = FAST_SDI_RGBA;
		alphaPadding = FAST_RGBA_ALPHA_PADDING_ZERO;
	} else if (strcmp(sdiFormat, "BMR10L") == 0) {
		SDIFormat = FAST_SDI_RGB_10_BMR10L;
	} else if (strcmp(sdiFormat, "BMR10B") == 0) {
		SDIFormat = FAST_SDI_RGB_10_BMR10B;
	} else if (strcmp(sdiFormat, "BMR12B") == 0) {
		SDIFormat = FAST_SDI_RGB_12_BMR12B;
	} else if (strcmp(sdiFormat, "BMR12L") == 0) {
		SDIFormat = FAST_SDI_RGB_12_BMR12L;
	} else {
		fprintf(stderr, "Incorrect SDI format (%s)\n", sdiFormat);
		fprintf(stderr, "setup default format FAST_SDI_YV12_BT709 \n");
		SDIFormat = FAST_SDI_YV12_BT709;
		return true;
	}

	return true;
}

bool SDIConverterOptions::Parse(int argc, char *argv[]) {
	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), "format", &tmp);
	if (tmp != NULL) {
		if (!FormatParser(tmp)) {
			return false;
		}
	} else {
		fprintf(stderr, "-format parameter was not found\n");
		fprintf(stderr, "Setup default format FAST_SDI_YV12_BT709 \n");
		SDIFormat = FAST_SDI_YV12_BT709;
	}

	ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), "o2", &FileNameAlternate);

	IsExport = ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "export");
	IsGpu = ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "gpu");

	Width = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "width");
	Height = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "height");

	IsConvert12to16 = ParametersParser::CheckCmdLineFlag(argc, const_cast<const char**>(argv), "convert16");

	return true;
}
