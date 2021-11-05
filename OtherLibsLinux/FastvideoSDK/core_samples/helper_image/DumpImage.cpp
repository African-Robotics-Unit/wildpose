/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "fastvideo_sdk.h"

#include <cstdio>

#include "SurfaceTraits.hpp"

template<typename T>
int DumpImageToText(
	fastSurfaceFormat_t surfaceFmt,

	T *img,
	const unsigned pitch,

	const unsigned width,
	const unsigned height,

	const char *fname
) {
	FILE *fp = fopen(fname, "w+");
	if (!fp) {
		fprintf(stderr, "Can not create output file\n");
		return -1;
	}

	for (unsigned i = 0; i < height; i++) {
		for (unsigned j = 0; j < width * GetNumberOfChannelsFromSurface(surfaceFmt); j++) {
			fprintf(fp, "%10d", img[i * pitch + j]);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
	return 0;
}

template<typename T>
int DumpImageChannelToText(
	T *img,
	const unsigned pitch,

	const unsigned width,
	const unsigned height,

	const char *fnameR,
	const char *fnameG,
	const char *fnameB
) {
	FILE *fpR = fopen(fnameR, "w+");
	FILE *fpG = fopen(fnameG, "w+");
	FILE *fpB = fopen(fnameB, "w+");
	if (!fpR || !fpG || !fpB) {
		fprintf(stderr, "Can not create output file\n");
		return -1;
	}

	for (unsigned i = 0; i < height; i++) {
		for (unsigned j = 0; j < width; j++) {
			fprintf(fpR, "%10d", img[i * pitch + j * 3 + 0]);
			fprintf(fpG, "%10d", img[i * pitch + j * 3 + 1]);
			fprintf(fpB, "%10d", img[i * pitch + j * 3 + 2]);
		}
		fprintf(fpR, "\n");
		fprintf(fpG, "\n");
		fprintf(fpB, "\n");
	}

	fclose(fpR);
	fclose(fpG);
	fclose(fpB);
	return 0;
}

template int DumpImageChannelToText<unsigned short>(
	unsigned short *img,
	const unsigned pitch,

	const unsigned width,
	const unsigned height,

	const char *fnameR,
	const char *fnameG,
	const char *fnameB
);