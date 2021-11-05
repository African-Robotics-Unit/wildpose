/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "ppm.h"
#include "helper_common.h"

#include <cmath>
#include <ctype.h>
#include <fstream>
#include <string.h>

#include "alignment.hpp"

int LoadHeaderPPM(
	const char* fname,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel
) {
	FILE *fp = NULL;

	if (FOPEN_FAIL(FOPEN(fp, fname, "rb")))
		return 0;

	unsigned int startPosition = 0;

	// check header
	char header[PGMHeaderSize] = { 0 };

	while (header[startPosition] == 0) {
		startPosition = 0;
		if (fgets(header, PGMHeaderSize, fp) == NULL)
			return 0;

		while (isspace(header[startPosition])) startPosition++;
	}

	if (strncmp(&header[startPosition], "P5", 2) == 0) {
		numberOfChannels = 1;
	} else if (strncmp(&header[startPosition], "P6", 2) == 0) {
		numberOfChannels = 3;
	} else {
		numberOfChannels = 0;
		return 0;
	}

	// parse header, read maxval, width and height
	unsigned int maxval = 0;
	unsigned int i = 0;
	unsigned int readsCount = 0;

	if ((i = SSCANF(&header[startPosition + 2], "%u %u %u", &width, &height, &maxval)) == EOF)
		i = 0;

	while (i < 3) {
		if (fgets(header, PGMHeaderSize, fp) == NULL)
			return 0;

		if (header[0] == '#')
			continue;

		if (i == 0) {
			if ((readsCount = SSCANF(header, "%u %u %u", &width, &height, &maxval)) != EOF)
				i += readsCount;
		} else if (i == 1) {
			if ((readsCount = SSCANF(header, "%u %u", &height, &maxval)) != (int)EOF)
				i += readsCount;
		} else if (i == 2) {
			if ((readsCount = SSCANF(header, "%u", &maxval)) != (int)EOF)
				i += readsCount;
		}
	}

	bitsPerChannel = int(log(maxval) / log(2) + 1);

	fclose(fp);
	return 1;
}

int LoadPPM(
	const char *file,
	void** data,
	BaseAllocator *alloc,
	unsigned int &width,
	unsigned &wPitch,
	unsigned int &height,
	unsigned &bitsPerChannel,
	unsigned &channels
) {
	FILE *fp = NULL;

	if (FOPEN_FAIL(FOPEN(fp, file, "rb")))
		return 0;

	unsigned int startPosition = 0;

	// check header
	char header[PGMHeaderSize] = { 0 };
	while (header[startPosition] == 0) {
		startPosition = 0;
		if (fgets(header, PGMHeaderSize, fp) == NULL)
			return 0;

		while (isspace(header[startPosition])) startPosition++;
	}

	bool textMode = false;
	bool fvHeader = false;
	int strOffset = 2;
	if (strncmp(&header[startPosition], "P5", 2) == 0) {
		channels = 1;
	} else if (strncmp(&header[startPosition], "P2", 2) == 0) {
		channels = 1;
		textMode = true;
	} else if (strncmp(&header[startPosition], "P6", 2) == 0) {
		channels = 3;
	} else if (strncmp(&header[startPosition], "P15", 3) == 0) {     //fv
		channels = 1;
		strOffset = 3;
		fvHeader = true;
	} else if (strncmp(&header[startPosition], "P16", 3) == 0) {    //fv
		channels = 3;
		strOffset = 3;
		fvHeader = true;
	} else {
		channels = 0;
		return 1;
	}

	// parse header, read maxval, width and height
	unsigned int maxval = 0;
	unsigned int i = 0;
	int readsCount = 0;

	if ((i = SSCANF(&header[startPosition + strOffset], "%u %u %u", &width, &height, &maxval)) == EOF)
		i = 0;

	while (i < 3) {
		if (fgets(header, PGMHeaderSize, fp) == NULL)
			return 0;

		if (header[0] == '#')
			continue;

		if (i == 0) {
			if ((readsCount = SSCANF(header, "%u %u %u", &width, &height, &maxval)) != EOF)
				i += readsCount;
		} else if (i == 1) {
			if ((readsCount = SSCANF(header, "%u %u", &height, &maxval)) != EOF)
				i += readsCount;
		} else if (i == 2) {
			if ((readsCount = SSCANF(header, "%u", &maxval)) != EOF)
				i += readsCount;
		}
	}

	bitsPerChannel = int(log(maxval) / log(2) + 1);
	const unsigned bytePerPixel = _uSnapUp<unsigned>(bitsPerChannel, 8) / 8;

	wPitch = GetAlignedPitch(width, channels, bytePerPixel, alloc->getAlignment());

	*data = alloc->allocate(wPitch * height);

	if (textMode) {
		unsigned short *d = static_cast<unsigned short *>(*data);
		for (i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				fscanf(fp, "%d", &d[i * (wPitch >> 1) + j]);
			}
		}
	} else {
		unsigned char *d = static_cast<unsigned char *>(*data);
		for (i = 0; i < height; i++) {
			if (fread(&d[i * wPitch], sizeof(unsigned char), width * bytePerPixel * channels, fp) == 0)
				return 0;

			if (bytePerPixel == 2 && !fvHeader) {
				unsigned short *p = reinterpret_cast<unsigned short *>(&d[i * wPitch]);
				for (unsigned int x = 0; x < wPitch / bytePerPixel; x++) {
					unsigned short t = p[x];
					const unsigned short t1 = t >> 8;
					t = (t << 8) | t1;
					p[x] = t;
				}
			}
		}
	}
	
	fclose(fp);
	return 1;
}

int SavePPM(
	const char *file,
	unsigned char *data,
	const unsigned w,
	const unsigned wPitch,
	const unsigned h,
	const int bitsPerChannel,
	const unsigned int channels
) {
	assert(NULL != data);
	assert(w > 0);
	assert(h > 0);

	std::fstream fh(file, std::fstream::out | std::fstream::binary);
	if (fh.bad())
		return 0;

	if (channels == 1) {
		fh << "P5\n";
	} else if (channels == 3) {
		fh << "P6\n";
	} else
		return 0;

	fh << w << "\n" << h << "\n" << ((1 << bitsPerChannel) - 1) << std::endl;
	const int bytePerPixel = _uSnapUp<unsigned>(bitsPerChannel, 8) / 8;

	unsigned short * tmp = (unsigned short *)malloc(wPitch);
	for (unsigned int y = 0; y < h && fh.good(); y++) {
		if (bytePerPixel == 2) {
			unsigned short *p = (unsigned short*)data;
			for (unsigned int x = 0; x < wPitch / bytePerPixel; x++) {
				unsigned short t = p[y * wPitch / bytePerPixel + x];
				unsigned short t1 = t >> 8;
				t = (t << 8) | t1;
				tmp[x] = t;
			}
			fh.write(reinterpret_cast<const char *>(tmp), w * channels * bytePerPixel);
		} else
			fh.write(reinterpret_cast<const char *>(&data[y * wPitch]), w * channels * bytePerPixel);
	}

	free(tmp);

	fh.flush();
	if (fh.bad())
		return 0;

	fh.close();
	return 1;
}