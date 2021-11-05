/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "BaseColorCorrectionOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>

bool BaseColorCorrectionOptions::Parse(int argc, char *argv[]) {
	BaseColorCorrectionEnabled = false;

	std::string colorCorrection("colorCorrection");
	if (OptionIndex != 0) {
		colorCorrection += "_" + std::to_string(OptionIndex);
	}

	if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), colorCorrection.c_str())) {
		BaseColorCorrectionEnabled = true;

		char *tmp = NULL;
		ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), colorCorrection.c_str(), &tmp);

		if (sscanf(tmp,
			"%*c%f%*c%f%*c%f%*c%f%*c%f%*c%f%*c%f%*c%f%*c%f%*c%f%*c%f%*c%f%*c",
			&BaseColorCorrection[0],
			&BaseColorCorrection[1],
			&BaseColorCorrection[2],
			&BaseColorCorrection[3],
			&BaseColorCorrection[4],
			&BaseColorCorrection[5],
			&BaseColorCorrection[6],
			&BaseColorCorrection[7],
			&BaseColorCorrection[8],
			&BaseColorCorrection[9],
			&BaseColorCorrection[10],
			&BaseColorCorrection[11]
		) < 12) {
			fprintf(stderr, "Incorrect -%s option (-%s %s)\n", colorCorrection.c_str(), colorCorrection.c_str(), tmp);
			BaseColorCorrectionEnabled = false;
		}
	}

	WhiteLevelEnabled = false;
	if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "whiteLevel")) {
		WhiteLevelEnabled = true;

		char *tmp = NULL;
		ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "whiteLevel", &tmp);

		if (sscanf(tmp,
			"%*c%d%*c%d%*c%d%*c",
			&WhiteLevel[0],
			&WhiteLevel[1],
			&WhiteLevel[2]
		) < 3) {
			fprintf(stderr, "Incorrect -whiteLevel option (-whiteLevel %s)\n", tmp);
			WhiteLevelEnabled = false;
		}
	}
	
	return true;
}
