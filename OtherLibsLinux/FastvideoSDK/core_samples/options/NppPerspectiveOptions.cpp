/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "NppPerspectiveOptions.h"
#include <string>

bool NppPerspectiveOptions::Parse(int argc, char *argv[]) {
	std::string optionName("perspectiveCoeffs");
	

	if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), optionName.c_str())) {

		char *tmp = NULL;
		ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), optionName.c_str(), &tmp);

		if (sscanf(tmp,
			"%*c%lf%*c%lf%*c%lf%*c%lf%*c%lf%*c%lf%*c%lf%*c%lf%*c%lf%*c",
			&Coeffs[0][0],
			&Coeffs[0][1],
			&Coeffs[0][2],
			&Coeffs[1][0],
			&Coeffs[1][1],
			&Coeffs[1][2],
			&Coeffs[2][0],
			&Coeffs[2][1],
			&Coeffs[2][2]
			) < 9) {
			fprintf(stderr, "Incorrect -%s option (-%s %s)\n", optionName.c_str(), optionName.c_str(), tmp);
		}
	}
	else
	{
		Coeffs[0][0] = Coeffs[1][1] = Coeffs[2][2] = 1.;
		Coeffs[0][1] = Coeffs[0][2] = 0.;
		Coeffs[1][0] = Coeffs[1][2] = 0.;
		Coeffs[2][0] = Coeffs[2][1] = 0.;
	}
	return true;
}