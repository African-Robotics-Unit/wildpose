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

#include "CropOptions.h"
#include "ParametersParser.h"

bool CropOptions::Parse(int argc, char *argv[]) {
	IsEnabled = true;

	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, OptionName, &tmp);
	if (tmp != NULL) {
		// 375x375+63+0
		if (sscanf(tmp, "%d%*c%d%*c%d%*c%d", &CropWidth, &CropHeight, &CropLeftTopCoordsX, &CropLeftTopCoordsY) < 4) {
			fprintf(stderr, "Incorrect -%s option (-%s %s)\n", OptionName, OptionName, tmp);
			IsEnabled = false;
		}
	} else {
		IsEnabled = false;
	}

	return true;
}
