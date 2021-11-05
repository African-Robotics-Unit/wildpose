/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include <thread>

#include "J2kDecoderOptions.h"
#include "ParametersParser.h"

void J2kDecoderOptions::Init() {
	ResolutionLevels = 0;
    DecodePasses = 0;
    MaxMemoryAvailable = 0;
    EnableMemoryReallocation = false;
    ForceTo8bits = false;

	Discard = false;
    Timeout = 0.0;
	

	MaxTileWidth = MaxTileHeight = 0;
    SequentialTiles = false;

	WindowLeftTopCoordsX = WindowLeftTopCoordsY = WindowWidth = WindowHeight = 0;
	IsEnabledWindow = false;

	Parsing = true; // opposite to "Truncation mode"
	TruncationRate = 0;
	TruncationLength = 0;
	EnableROI = false;

	DecodePasses = 0;
	ForceTo8bits = false;
	Timeout = 0;
	PrintGML = false;
	OutputPathGML = nullptr;
}

J2kDecoderOptions::J2kDecoderOptions():BaseOptions() {
	// Set default values
	Init();
}

J2kDecoderOptions::J2kDecoderOptions(bool ignoreOutput) : BaseOptions(ignoreOutput, false) {
	Init();
}

bool J2kDecoderOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

    const int maxResolutionLevels = 12;

	ResolutionLevels = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "l");
    if (ResolutionLevels < 0 || ResolutionLevels > maxResolutionLevels) {
		if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "l"))
			printf("The specified number of resolution levels is out of range. Setting it to default.\n");
        ResolutionLevels = 0;
	}

	DecodePasses = 0;
	unsigned DecodeBits = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "bits");
	if (DecodeBits < 0 || DecodeBits > 16)
	{
		if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "bits"))
			printf("The specified number of bitplanes is out of range. Setting it to default.\n");
		DecodePasses = 0;
	}
	else if (DecodeBits > 0)
	{
		DecodePasses = 1 + (DecodeBits - 1) * 3;
	}

	unsigned DecodePasses_ = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "passes");

	if (DecodePasses_ < 0 || DecodePasses_ > 48)
	{
		if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "passes"))
            printf("The specified number of passes is out of range. Setting it to default.\n");
        DecodePasses = 0;
	}
	else if (DecodePasses_ > 0)
	{
		DecodePasses = DecodePasses_;
	}
		
	int memory = (size_t)ParametersParser::GetCmdLineArgumentInt(argc, (const char**)argv, "memoryLimit");
    if (memory < 0)
	{
        if (ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "memoryLimit"))
			printf("The specified amount of memory is out of range. Setting it to default.\n");
		MaxMemoryAvailable = 0;
	}
	else
	{
		MaxMemoryAvailable = (size_t)memory * (1024 * 1024);
	}

	EnableMemoryReallocation = ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "dynamicAllocation");
	
	ForceTo8bits = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "forceTo8bits");
	Discard = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "discard");
 	Timeout = ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "timeout");
    if (Timeout < 0.0)
    {
        if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "timeout"))
            printf("The specified timeout value is invalid. Setting it to default.\n");
        Timeout = 0.0;
    }
	Tier2Threads = ParametersParser::GetCmdLineArgumentInt(argc, (const char**)argv, "t2thread");	
	MaxTileWidth = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "maxTileWidth", 0);
	MaxTileHeight = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "maxTileHeight", 0);
	SequentialTiles = ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "sequentialTiles");
	
    char* tmp = nullptr;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "window", &tmp);
    if (tmp != nullptr) {
		IsEnabledWindow = true;

		// 375x375+63+0
		if (
			sscanf(tmp, "%d%*c%d%*c%d%*c%d", &WindowWidth, &WindowHeight, &WindowLeftTopCoordsX, &WindowLeftTopCoordsY) < 4 ||
			WindowWidth < 0 || WindowHeight < 0 || WindowLeftTopCoordsX < 0 || WindowLeftTopCoordsY < 0
		) {
			fprintf(stderr, "Incorrect -window option (-window %s)\n", tmp);
			IsEnabledWindow = false;
		}
    }
    else {
		IsEnabledWindow = false;
	}
	
	PrintBoxes = ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "printboxes");

    PrintGML = ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "printgml");

	ParametersParser::GetCmdLineArgumentString(argc, (const char**)argv, "outputgml", &OutputPathGML);

	return true;
}
