/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "ResizerOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <cstring>

bool ResizerOptions::Parse(int argc, char *argv[]) {
	OutputWidth = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "outputWidth");
	if (argc > 1 && OutputWidth < MIN_SCALED_SIZE) {
		fprintf(stderr, "Unsupported output image width - %d. Minimum width is %d\n", OutputWidth, MIN_SCALED_SIZE);
		return false;
	}

	OutputHeight = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "outputHeight");
	OutputHeightEnabled = OutputHeight > 0;

	BackgroundEnabled = false;
	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "background", &tmp);
	if (tmp != NULL) {
		// 375,375,63
		const int count = sscanf(tmp, "%d%*c%d%*c%d", &(Background[0]), &(Background[1]), &(Background[2]));
		if (count != 3 && count != 1) {
			fprintf(stderr, "Incorrect -background option (-background %s)\n", tmp);
		} else {
			BackgroundEnabled = true;
		}
	}

	ShiftX = 0.0f;
	ShiftY = 0.0f;

	char *modeStr = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "mode", &modeStr);
	if (modeStr != NULL) {
		if (strcmp(modeStr, "super") == 0) {
			Mode = SUPER;
		} else if (strcmp(modeStr, "lanczos") == 0) {
			Mode = LANCZOS;
		} else if (strcmp(modeStr, "fastvideo") == 0) {
			Mode = FASTVIDEO;
		} else {
			Mode = FASTVIDEO;
		}
	} else {
		Mode = FASTVIDEO;
	}

	return true;
}
