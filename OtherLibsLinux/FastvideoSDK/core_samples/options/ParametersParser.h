/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __PARAMETERS_PARSER__
#define __PARAMETERS_PARSER__

class ParametersParser
{
private:
	static int StringRemoveDelimiter(char delimiter, const char *string);

public:
	ParametersParser(void) { };
	~ParametersParser(void) { };

	static int GetFileExtension(char *filename, char **extension);
	static bool CheckCmdLineFlag(const int argc, const char **argv, const char *string_ref);
	static bool GetCmdLineArgumentString(const int argc, const char **argv, const char *string_ref, char **string_retval);
	static int GetCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref);
	static int GetCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref, const int NotFoundValue);
	static double GetCmdLineArgumentFloat(const int argc, const char **argv, const char *string_ref, const double NotFoundValue = 0.0f);
};

#endif // __PARAMETERS_PARSER__