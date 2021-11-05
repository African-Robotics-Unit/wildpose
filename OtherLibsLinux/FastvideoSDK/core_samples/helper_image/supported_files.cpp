 /*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <cstdio>
#include <string>

#include <sstream>

#include "helper_raw.hpp"
#include "helper_dir.hpp"
#include "supported_files.hpp"

#include "FastAllocator.h"
#include "checks.h"

std::string generateOutputFileName(const char *pattern, unsigned idx) {
	std::string outputFileName(pattern);
	std::string::size_type loc = outputFileName.find("*", 0);
	if (loc != std::string::npos) {
		std::ostringstream convert;
		convert << idx;
		outputFileName.replace(loc, 1, convert.str());
	}
	return outputFileName;
}

template<typename T>
void CreateListForThread(std::list<T> &src, std::list<T> &dst, int index) {
	for (auto i = src.begin(); i != src.end(); i++) {
		T img = (*i);

		size_t found = img.outputFileName.rfind(".");
		if (found != std::string::npos) {
			img.outputFileName.replace(found, 1, "_" + std::to_string(index) + ".");
		}

		dst.push_back(img);
	}
}

bool IsGrayUncompressedFormat(const char* fname) {
	if (compareFileExtension(fname, ".pgm") || compareFileExtension(fname, ".tif") || compareFileExtension(fname, ".tiff") || compareFileExtension(fname, ".bmp"))
		return true;
	return false;
}

bool IsColorUncompressedFormat(const char* fname) {
	if (compareFileExtension(fname, ".ppm") || compareFileExtension(fname, ".tif") || compareFileExtension(fname, ".tiff") || compareFileExtension(fname, ".bmp"))
		return true;

	return false;
}

template void CreateListForThread(std::list<Bytestream<FastAllocator> > &src, std::list<Bytestream<FastAllocator>> &dst, int index);
template void CreateListForThread(std::list<Image<FastAllocator> >& src, std::list<Image<FastAllocator>>& dst, int index);
