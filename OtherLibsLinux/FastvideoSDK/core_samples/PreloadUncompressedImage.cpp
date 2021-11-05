/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "PreloadUncompressedImage.hpp"

#include <list>
#include <string>

#include "helper_dir.hpp"
#include "bmp.hpp"
#include "ppm.h"
#include "tiff.h"
#include "checks.h"

fastStatus_t PreloadImageFromFile(
	const char* fname,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel
) {
	if (compareFileExtension(fname, ".pgm") ||
		compareFileExtension(fname, ".ppm")) {
		LoadHeaderPPM(fname, width, height, numberOfChannels, bitsPerChannel);
	} else if (compareFileExtension(fname, ".bmp")) {
		LoadHeaderBMP(fname, width, height, numberOfChannels, bitsPerChannel);
	} else if (compareFileExtension(fname, ".tif") ||
		compareFileExtension(fname, ".tiff")) {
		LoadHeaderTIFF(fname, width, height, numberOfChannels, bitsPerChannel);
	}

	return FAST_OK;
}

fastStatus_t PreloadImageFromFolder(
	const char* folderName,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel
) {
	std::list<std::string> files;
	{
		getFileList(folderName, files);
		if (files.empty()) {
			fprintf(stderr, "No input files found\n");
			return FAST_IO_ERROR;
		}
	}

	CHECK_FAST(PreloadImageFromFile(
		files.begin()->c_str(),
		width, height, numberOfChannels, bitsPerChannel
	));

	files.clear();
	return FAST_OK;
}


fastStatus_t PreloadImage(const char* path,
	bool isFolder,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel)
{
	return isFolder ? PreloadImageFromFolder(path, width, height, numberOfChannels, bitsPerChannel) : PreloadImageFromFile(path, width, height, numberOfChannels, bitsPerChannel);
}
