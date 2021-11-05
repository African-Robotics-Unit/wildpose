/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "ParametersParser.h"

#include <string>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#ifndef STRCASECMP
#define STRCASECMP  _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#else
#ifndef STRCASECMP
#define STRCASECMP  strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#endif

int ParametersParser::StringRemoveDelimiter(char delimiter, const char *string) {
	int string_start = 0;

	while (string[string_start] == delimiter) {
		string_start++;
	}

	if (string_start > (int)strlen(string) - 1) {
		return 0;
	}

	return string_start;
}

int ParametersParser::GetFileExtension(char *filename, char **extension) {
	int string_length = (int)strlen(filename);

	while (filename[string_length--] != '.') {
		if (string_length == 0)
			break;
	}
	if (string_length > 0) string_length += 2;

	if (string_length == 0) 
		*extension = NULL;
	else 
		*extension = &filename[string_length];

	return string_length;
}


bool ParametersParser::CheckCmdLineFlag(const int argc, const char **argv, const char *string_ref) {
	bool bFound = false;

	if (argc >= 1) {
		for (int i=1; i < argc; i++) {
			int string_start = StringRemoveDelimiter('-', argv[i]);
			const char *string_argv = &argv[i][string_start];

			const char *equal_pos = strchr(string_argv, '=');
			int argv_length = (int)(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

			int length = (int)strlen(string_ref);

			if (length == argv_length && !STRNCASECMP(string_argv, string_ref, length)) {
				bFound = true;
				break;
			}
		}
	}

	return bFound;
}

bool ParametersParser::GetCmdLineArgumentString(const int argc, const char **argv, const char *string_ref, char **string_retval) {
	bool bFound = false;

	if (argc >= 1) {
		for (int i=1; i < argc; i++) {
			int string_start = StringRemoveDelimiter('-', argv[i]);
			char *string_argv = (char *)&argv[i][string_start];
			unsigned length = (int)strlen(string_ref);
			
			if (length != strlen(string_argv))
				continue;

			if (!STRNCASECMP(string_argv, string_ref, length)) {
				if (i + 1 < argc) {
					*string_retval = (char *)argv[i+1];
					length = (int)strlen(*string_retval);
					while(isspace((*string_retval)[length - 1])) {
						(*string_retval)[length - 1] = '\0';
						length--;
					}

					bFound = true;
					break;
				}
			}
		}
	}

	if (!bFound) {
		*string_retval = NULL;
	}

	return bFound;
}

int ParametersParser::GetCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref) {
	char *foundValue = NULL;
	GetCmdLineArgumentString(argc, argv, string_ref, &foundValue);
	return (foundValue == NULL) ? 0 : atoi(foundValue);
}

int ParametersParser::GetCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref, const int NotFoundValue) {
	char *foundValue = NULL;
	GetCmdLineArgumentString(argc, argv, string_ref, &foundValue);
	return (foundValue == NULL) ? NotFoundValue : atoi(foundValue);
}

double ParametersParser::GetCmdLineArgumentFloat(const int argc, const char **argv, const char *string_ref, const double NotFoundValue) {
	char *foundValue = NULL;
	GetCmdLineArgumentString(argc, argv, string_ref, &foundValue);
	if ( foundValue == NULL ) {
		return NotFoundValue;
	}
	double value = (double )atof(foundValue);
	if ( value == 0.0 && foundValue[0] != '0' ) {
		// foundValue is not number
		return NotFoundValue;
	}
	return value;
}
