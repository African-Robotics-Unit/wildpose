/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "helper_dir.hpp"
#include <fstream>
#include <cstring>
#include <string>
#include <algorithm>

#ifndef __GNUC__
#include <io.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <dirent.h>
#endif

bool compareFileExtension(const char *fileName, const char *etalon) {
	bool success = true;
	unsigned int startPosition = static_cast<unsigned int>(strlen(fileName));
	while (fileName[startPosition] != '.' && startPosition > 0) startPosition--;

	for (unsigned i = 0; i < strlen(etalon) + 1; i++)
		if (tolower(fileName[startPosition + i]) != etalon[i]) {
			success = false;
			break;
		}
	return success;
}

bool compareMask(std::string &fileMask, std::string &fileName)
{
	int pos = fileMask.find('*');
	if (pos == std::string::npos)
		return false;
	if (fileMask == "*")
		return true;
	std::string prefix = fileMask.substr(0, static_cast<size_t>(pos));
	std::string suffix = fileMask.substr(static_cast<size_t>(pos + 1));

	bool prefixCond = prefix.empty() ? true : prefix.size() <= fileName.size() && fileName.substr(0, prefix.size()) == prefix;

	bool suffixCond = suffix.empty() ? true : suffix.size() <= fileName.size() && fileName.substr(fileName.size() - suffix.size(), suffix.size()) == suffix;
	
	return prefixCond && suffixCond;
}

bool fileExist(const char *fileName) {
#ifndef __GNUC__
	return _access(fileName, 4) != -1;
#else
	struct stat buffer;
	return (stat(fileName, &buffer) == 0);
#endif
}

int fileSize(const char *fname) {
	std::ifstream in(fname, std::ifstream::in | std::ifstream::binary);
	in.seekg(0, std::ifstream::end);
	return (int)in.tellg();
}

int getFileList(const char *folderName, std::list<std::string> &fileList) {
	fileList.clear();

	std::string strFolderPath(folderName);
	std::string fileMask (folderName);
	std::replace(strFolderPath.begin(), strFolderPath.end(), '\\', '/');
	int pos = strFolderPath.rfind('/');
	if (pos != std::string::npos) {
		strFolderPath = strFolderPath.substr(0, static_cast<size_t>(pos + 1));
		fileMask = fileMask.substr(static_cast<size_t>(pos + 1));
	} else
		strFolderPath.erase();

#ifndef __GNUC__
	std::string strPattern(folderName);
	WIN32_FIND_DATA FileInformation;			// File information

	const HANDLE hFile = ::FindFirstFile(strPattern.c_str(), &FileInformation);
	if (hFile != INVALID_HANDLE_VALUE) {
		do {
			if (FileInformation.cFileName[0] != '.') {
				if (!(FileInformation.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
					fileList.push_back(strFolderPath + "/" + FileInformation.cFileName);
				}
			}
		} while (::FindNextFile(hFile, &FileInformation) == TRUE);

		::FindClose(hFile);
	}

#else
	struct dirent *dir;
	DIR *d = opendir(strFolderPath.c_str());

	if (d) {
		while ((dir = readdir(d)) != NULL) {
			if (dir->d_type == DT_REG) {
				std::string fileName(dir->d_name);
				if (compareMask(fileMask, fileName)) {
					fileList.push_back(strFolderPath + "/" + std::string(dir->d_name));
				}
			}
		}
		closedir(d);
	}
#endif
	fileList.sort();
	return 1;
}