/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "LutOptions.h"
#include "ParametersParser.h"
#include "stdio.h"

bool LutOptions::Parse(int argc, char *argv[]) {
	Lut = Lut_R = Lut_G = Lut_B = NULL;

	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "lut", &Lut);
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "lut_r", &Lut_R);
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "lut_g", &Lut_G);
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "lut_b", &Lut_B);
	
	if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_12")) {
		ImageFilter = FAST_LUT_12_12;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_12c")) {
		ImageFilter = FAST_LUT_12_12_C;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_16")) {
		ImageFilter = FAST_LUT_12_16;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_16c")) {
		ImageFilter = FAST_LUT_12_16_C;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_8")) {
		ImageFilter = FAST_LUT_12_8;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_8c")) {
		ImageFilter = FAST_LUT_12_8_C;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_16")) {
		ImageFilter = FAST_LUT_16_16;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_16c")) {
		ImageFilter = FAST_LUT_16_16_C;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_16_fr")) {
		ImageFilter = FAST_LUT_16_16_FR;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_16c_fr")) {
		ImageFilter = FAST_LUT_16_16_FR_C;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_8")) {
		ImageFilter = FAST_LUT_16_8;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_8c")) {
		ImageFilter = FAST_LUT_16_8_C;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut8_16b")) {
		ImageFilter = FAST_LUT_8_16_BAYER;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut10_16b")) {
		ImageFilter = FAST_LUT_10_16_BAYER;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut12_16b")) {
		ImageFilter = FAST_LUT_12_16_BAYER;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut14_16b")) {
		ImageFilter = FAST_LUT_14_16_BAYER;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_16b")) {
		ImageFilter = FAST_LUT_16_16_BAYER;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "lut16_16b_fr")) {
		ImageFilter = FAST_LUT_16_16_FR_BAYER;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "hsvLut3D")) {
		ImageFilter = FAST_HSV_LUT_3D;
	} else if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "rgbLut3D")) {
		ImageFilter = FAST_RGB_LUT_3D;
	}

	lutSize1D = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "size1D");

	lutSize3D.x = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "sizeX");
	lutSize3D.y = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "sizeY");
	lutSize3D.z = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "sizeZ");
	/*if (lutSize3D.x == 0 || lutSize3D.y == 0 || lutSize3D.z == 0) {
		if (lutSize1D != 0) {
			lutSize3D.x = lutSize3D.y = lutSize3D.z = lutSize1D;
		}
	}*/

	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "sourceChannel", &tmp);
	if (tmp != NULL) {
		int channel_0, channel_1, channel_2;
		if (sscanf(tmp, "%d%*c%d%*c%d%*c", &channel_0, &channel_1, &channel_2) < 3) {
			fprintf(stderr, "Incorrect -sourceChannel option (-sourceChannel %s)\n", tmp);
		}

		if (channel_0 < 3 && channel_1 < 3 && channel_2 < 3) {
			ChannelType[0] = (fastColorSaturationChannelType_t)channel_0;
			ChannelType[1] = (fastColorSaturationChannelType_t)channel_1;
			ChannelType[2] = (fastColorSaturationChannelType_t)channel_2;
		} else {
			ChannelType[0] = FAST_CHANNEL_H;
			ChannelType[1] = FAST_CHANNEL_S;
			ChannelType[2] = FAST_CHANNEL_L_OR_V;
		}
	} else {
		ChannelType[0] = FAST_CHANNEL_H;
		ChannelType[1] = FAST_CHANNEL_S;
		ChannelType[2] = FAST_CHANNEL_L_OR_V;
	}

	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "operationType", &tmp);
	if (tmp != NULL) {
		int operation_0, operation_1, operation_2;
		if (sscanf(tmp, "%d%*c%d%*c%d%*c", &operation_0, &operation_1, &operation_2) < 3) {
			fprintf(stderr, "Incorrect -operationType option (-operationType %s)\n", tmp);
		}

		if (operation_0 < 3 && operation_1 < 3 && operation_2 < 3) {
			OperationType[0] = (fastColorSaturationOperationType_t)operation_0;
			OperationType[1] = (fastColorSaturationOperationType_t)operation_1;
			OperationType[2] = (fastColorSaturationOperationType_t)operation_2;
		} else {
			OperationType[0] = FAST_OP_REPLACE;
			OperationType[1] = FAST_OP_REPLACE;
			OperationType[2] = FAST_OP_REPLACE;
		}
	} else {
		OperationType[0] = FAST_OP_REPLACE;
		OperationType[1] = FAST_OP_REPLACE;
		OperationType[2] = FAST_OP_REPLACE;
	}

	return true;
}
