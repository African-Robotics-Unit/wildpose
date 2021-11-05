/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "WhiteBalanceOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>
#include <string.h>

bool WhiteBalanceOptions::Parse(int argc, char *argv[]) {
	R = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "wb_r", 1.0));
	G1 = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "wb_g1", 1.0));
	G2 = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "wb_g2", 1.0));
	B = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "wb_b", 1.0));

	IsEnabled = R != 1.0f || G1 != 1.0f || G2 != 1.0f || B != 1.0f;

	return true;
}
