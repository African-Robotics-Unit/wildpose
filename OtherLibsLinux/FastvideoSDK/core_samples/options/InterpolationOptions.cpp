/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "InterpolationOptions.h"
#include "ParametersParser.h"
#include "string.h"

fastNPPImageInterpolation_t ToInterpolationType(const char *str)  {
	if (strcmp(str, "linear") == 0) {
		return NPP_INTER_LINEAR;
	}
	else if (strcmp(str, "cubic") == 0) {
		return NPP_INTER_CUBIC;
	}
	else if (strcmp(str, "bspline") == 0) {
		return NPP_INTER_CUBIC2P_BSPLINE;
	}
	else if (strcmp(str, "catmullrom") == 0) {
		return NPP_INTER_CUBIC2P_CATMULLROM;
	}
	else if (strcmp(str, "b05c03") == 0) {
		return NPP_INTER_CUBIC2P_B05C03;
	}
	else if (strcmp(str, "super") == 0) {
		return NPP_INTER_SUPER;
	}
	else if (strcmp(str, "lanczos") == 0) {
		return NPP_INTER_LANCZOS;
	}
	return NPP_INTER_CUBIC;
}

bool NppInterpolationOptions::Parse(int argc, char *argv[]) {
	char *tmp = NULL;
	if (ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), "interpolationType", &tmp)) {
		Type = ToInterpolationType(tmp);
	}
	else {
		Type = NPP_INTER_CUBIC;
	}

	return true;
}
