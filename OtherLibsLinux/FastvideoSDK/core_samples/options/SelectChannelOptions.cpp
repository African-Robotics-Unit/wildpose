/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "SelectChannelOptions.h"
#include "ParametersParser.h"

#include <cstring>

fastChannelType_t SelectChannelOptions::GetChannelType(const char* str) {
	if (strcmp(str, "R") == 0)
		return FAST_CHANNEL_R;
	if (strcmp(str, "G") == 0)
		return FAST_CHANNEL_G;
	if (strcmp(str, "B") == 0)
		return FAST_CHANNEL_B;
	return FAST_CHANNEL_R;
}


bool SelectChannelOptions::Parse(int argc, char *argv[]) {
	char *tmp;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "channel", &tmp);

	Channel = GetChannelType(tmp);

	return true;
}
