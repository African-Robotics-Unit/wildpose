/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FilesParameterChecker_HPP__
#define __FilesParameterChecker_HPP__

class FilesParameterChecker {
public:
	typedef enum {
		FAST_RAW,
		FAST_RAW_COLOR,
		FAST_RAW_GRAY,
		FAST_BMP,
		FAST_JPEG,
		FAST_JPEG2000,
		FAST_AVI,
		FAST_MP4,
		FAST_MXF,
		FAST_YV12,
		FAST_SDI,
		FAST_GRAY_COLOR,
	} fastFileType_t;

	typedef enum {
		FAST_OK,
		FAST_INPUT_ERROR,
		FAST_OUTPUT_ERROR,
		FAST_BOTH_ERROR
	} fastFileStatus_t;

private:
	static bool IsRaw(const char *fileName);
	static bool IsBmp(const char *fileName);
	static bool IsRawColor(const char *fileName);
	static bool IsRawGray(const char *fileName);
	static bool IsJpeg(const char *fileName);
	static bool IsAvi(const char *fileName);
	static bool IsMxf(const char *fileName);
	static bool IsMp4(const char *fileName);
	static bool IsJPEG2000(const char *fileName);
	static bool IsYV12(const char *fileName);
	static bool IsSDI(const char *fileName);
	static bool IsGrayColor(const char *fileName);
	static bool IsValid(const char *fileName, const fastFileType_t fileType);

public:
	static bool CheckFileExtension(const char *fileName, const char *etalon);
	static fastFileStatus_t Validate(const char *inputFile, fastFileType_t inputFileType, const char *outputFile, fastFileType_t outputFileType);
	static fastFileStatus_t Validate(const char * inputFile, fastFileType_t  inputFileType);
};

#endif	// __FilesParameterChecker_HPP__
