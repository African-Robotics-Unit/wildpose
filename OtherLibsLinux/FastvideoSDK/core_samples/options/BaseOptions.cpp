/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "BaseOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>
#include <cstring>
#include <algorithm>

#include "checks.h"

bool BaseOptions::CheckFileExtension(const char *fileName, const char *etalon) {
	bool success = true;
	unsigned startPosition = static_cast<unsigned>(strlen(fileName));
	while (fileName[startPosition] != '.' && startPosition > 0) startPosition--;

	for (unsigned i = 0; i < strlen(etalon)+1; i++)
		if (tolower(fileName[startPosition + i]) != etalon[i]) {
			success = false;
			break;
		}
	return success;
}

fastSurfaceFormat_t BaseOptions::GetSurfaceFormatFromExtension(const char* fname) {
	if (CheckFileExtension(fname, ".bmp") || CheckFileExtension(fname, ".dib") || CheckFileExtension(fname, ".jpg") || CheckFileExtension(fname, ".jpeg"))
		return FAST_BGR8;
	if (CheckFileExtension(fname, ".ppm"))
		return FAST_RGB8;
	if (CheckFileExtension(fname, ".ppm16"))
		return FAST_RGB16;
	if (CheckFileExtension(fname, ".pgm16"))
		return FAST_I16;
	return FAST_I8;
}

bool BaseOptions::IsMustBeConverted(const char *fname) {
	if (CheckFileExtension(fname, ".bmp") || CheckFileExtension(fname, ".jpg")) {
		return true;
	}

	return false;
}

bool BaseOptions::Parse(int argc, char *argv[]) {
	Help = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "help") || argc == 1;
	if (Help) {
		return true;
	}

	MultiProcess = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "multiprocess");
	AsyncMode = ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "async");

	InputPath = OutputPath = NULL;
	if (!ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "i", &InputPath)) {
		if (!ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "if", &InputPath)) {
			fprintf(stderr, "Input path parameter is not found\n");
			return false;
		}
	}

	if (!IgnoreOutput) {
		if (!ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "o", &OutputPath)) {
			fprintf(stderr, "Output file parameter is not found\n");
			return false;
		}
		ConvertToBGR = IsMustBeConverted(OutputPath);
	}

	IsFolder = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "if");
	MaxWidth = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "maxWidth");
	MaxHeight = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "maxHeight");

	RawWidth = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "w");
	RawHeight = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "h");

	if (RawWidth > 0)
		MaxWidth = std::max(RawWidth, MaxWidth);

	if (RawHeight > 0)
		MaxHeight = std::max(RawHeight, MaxHeight);

	if (IsFolder && !IgnoreMaxDimension) {
		if (MaxWidth <= 0 || MaxHeight <= 0) {
			fprintf(stderr, "maxWidth or maxHeight option was not set\n");
			return false;
		}
	}

	RepeatCount = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "repeat", 1);
	if (RepeatCount <= 0) RepeatCount = 1;

	NumberOfThreads = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "thread", 1);
	if (NumberOfThreads <= 1) NumberOfThreads = 1;

	NumberOfReaderThreads = ParametersParser::GetCmdLineArgumentInt(argc, (const char**)argv, "threadR", 1);
	if (NumberOfReaderThreads <= 1) NumberOfReaderThreads = 1;

	NumberOfWriterThreads = ParametersParser::GetCmdLineArgumentInt(argc, (const char**)argv, "threadW", 1);
	if (NumberOfWriterThreads <= 1) NumberOfWriterThreads = 1;

	BitsPerChannel = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "bits");
	if (BitsPerChannel != 8 && BitsPerChannel != 12 && BitsPerChannel != 16 && BitsPerChannel != 10 && BitsPerChannel != 14) {
		BitsPerChannel = 8;
	}

	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "lut", &Lut);

	DeviceId = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "d");
	SurfaceFmt = GetSurfaceFormatFromExtension(InputPath);

	ExclusiveMode = ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "exclusive");

	Info = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "info");
	
	BenchmarkInfo = ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "binfo");

	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "log", &LogFile);

	Discard = ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "discard");

	BatchSize = ParametersParser::GetCmdLineArgumentInt(argc, (const char**)argv, "b", 1);
	if (BatchSize < 1) {
		if (ParametersParser::CheckCmdLineFlag(argc, (const char**)argv, "b"))
			printf("The specified batch size is invalid. Setting it to default.\n");
		BatchSize = 1;
	}

	Queue = ParametersParser::GetCmdLineArgumentInt(argc, (const char**)argv, "queue", 16);
	if (Queue < 1) {
		Queue = 16;
	}

	return true;
}
