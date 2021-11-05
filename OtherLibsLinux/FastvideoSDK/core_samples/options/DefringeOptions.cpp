/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "DefringeOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>
#include <vector>
#include <sstream>

bool DefringeOptions::Parse(int argc, char *argv[]) {
	WindowSize = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "window");
	if (WindowSize <= 0 || WindowSize > 40) {
		fprintf(stderr, "Unsupported window size\n");
		return false;
	}

	TintR = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "tintR");
	TintG = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "tintG");
	TintB = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "tintB");

	Fi_tint = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "fi_tint"));
	if (Fi_tint == 0) {
		Fi_tint = -190;
	}

	Fi_max = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "fi_max"));
	if (Fi_max == 0) {
		Fi_max = 60;
	}

	Coefficient = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "coefficient"));
	if (Coefficient == 0.f) {
		Coefficient = 0.1f;
	}

	return true;
}
