/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __BASE_OPTIONS__
#define __BASE_OPTIONS__

#include "fastvideo_sdk.h"

class BaseOptions {
	bool IsMustBeConverted(const char *fname);

public:
	char *InputPath;
	char *OutputPath;
	bool IsFolder;

	bool AsyncMode;

	unsigned RawWidth;
	unsigned RawHeight;

	unsigned BitsPerChannel;
	char *Lut;

	bool Info;
	bool BenchmarkInfo;
	bool Help;
	bool ConvertToBGR;
	bool ExclusiveMode;

	unsigned DeviceId;

	unsigned MaxWidth;
	unsigned MaxHeight;

	fastSurfaceFormat_t SurfaceFmt;

	unsigned RepeatCount;
	unsigned NumberOfThreads;

	unsigned NumberOfReaderThreads;
	unsigned NumberOfWriterThreads;

	bool MultiProcess;

	char *LogFile;

	bool IgnoreOutput;
	bool IgnoreMaxDimension;

	bool Discard;
	unsigned BatchSize;
	unsigned Queue;

	
	BaseOptions(void) {
		InputPath = nullptr;
		OutputPath = nullptr;
		IsFolder = false;

		RawWidth = RawHeight = 0;

		BitsPerChannel = 0;
		Lut = nullptr;

		Info = false;
		BenchmarkInfo = false;
		Help = false;
		ConvertToBGR = false;
		ExclusiveMode = false;

		DeviceId = 0;

		MaxWidth = MaxHeight = 0;

		SurfaceFmt = FAST_I8;

		RepeatCount = 0;
		NumberOfThreads = 1;
		NumberOfReaderThreads = 1;
		NumberOfWriterThreads = 1;

		MultiProcess = false;

		IgnoreOutput = false;
		IgnoreMaxDimension = false;
		ConvertToBGR = false;

		LogFile = nullptr;

		Discard = false;
		BatchSize = 1;
		Queue = 16;
	}

	BaseOptions(bool ignoreOutput, bool ignoreMaxDimension) : BaseOptions() {
		IgnoreOutput = ignoreOutput;
		IgnoreMaxDimension = ignoreMaxDimension;
	}

	~BaseOptions(void) {
	}

	virtual bool Parse(int argc, char *argv[]);

	static bool CheckFileExtension(const char *fileName, const char *etalon);
	static fastSurfaceFormat_t GetSurfaceFormatFromExtension(const char* fname);
};

#endif // __BASE_OPTIONS__