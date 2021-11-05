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

#include "FastAllocator.h"
#include "checks.h"
#include "Image.h"

#include "IdentifySurface.hpp"
#include "SurfaceTraits.hpp"
#include "supported_files.hpp"
#include "ParametersParser.h"

fastStatus_t MosaicRGGB(Image<FastAllocator> &inputImg, Image<FastAllocator> &outputImg) {
	outputImg.w = inputImg.w;
	outputImg.h = inputImg.h;
	outputImg.wPitch = inputImg.w;
	outputImg.surfaceFmt = IdentifySurface(GetBitsPerChannelFromSurface(inputImg.surfaceFmt), 1);
	outputImg.samplingFmt = FAST_JPEG_Y;

	const unsigned bytePerChannel = GetBytesPerChannelFromSurface(inputImg.surfaceFmt);

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(outputImg.data.reset((unsigned char *)alloc.allocate(inputImg.w*inputImg.h*bytePerChannel)));

	const unsigned w = inputImg.w;
	const unsigned h = inputImg.h;
	if (bytePerChannel == 1) {
		for (unsigned i = 0; i < h; i++) {
			for (unsigned j = 0; j < w; j++) {
				unsigned char *c = &inputImg.data.get()[w * i * 3 + j * 3];

				if (i % 2 == 0 && j % 2 == 0)
					outputImg.data.get()[w * i + j] = *c;
				else if (i % 2 == 1 && j % 2 == 1)
					outputImg.data.get()[w * i + j] = c[2];
				else
					outputImg.data.get()[w * i + j] = c[1];
			}
		}
	} else if (bytePerChannel == 2) {
		unsigned short *data = reinterpret_cast<unsigned short *>(inputImg.data.get());
		unsigned short *odata = reinterpret_cast<unsigned short *>(outputImg.data.get());
		for (unsigned i = 0; i < h; i++) {
			for (unsigned j = 0; j < w; j++) {
				unsigned short *c = &data[w * i * 3 + j * 3];

				if (i % 2 == 0 && j % 2 == 0)
					odata[w * i + j] = *c;
				else if (i % 2 == 1 && j % 2 == 1)
					odata[w * i + j] = c[2];
				else
					odata[w * i + j] = c[1];
			}
		}
	}
	return FAST_OK;
}

fastStatus_t MosaicGBRG(Image<FastAllocator> &inputImg, Image<FastAllocator> &outputImg) {
	outputImg.w = inputImg.w;
	outputImg.h = inputImg.h;
	outputImg.wPitch = inputImg.w;
	outputImg.surfaceFmt = IdentifySurface(GetBitsPerChannelFromSurface(inputImg.surfaceFmt), 1);
	outputImg.samplingFmt = FAST_JPEG_Y;

	const unsigned bytePerChannel = GetBytesPerChannelFromSurface(inputImg.surfaceFmt);

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(outputImg.data.reset((unsigned char *)alloc.allocate(inputImg.w*inputImg.h *bytePerChannel)));

	const unsigned w = inputImg.w;
	const unsigned h = inputImg.h;
	if (bytePerChannel == 1) {
		for (unsigned i = 0; i < h; i++) {
			for (unsigned j = 0; j < w; j++) {
				unsigned char *c = &inputImg.data.get()[(w * i + j) * 3];
				if (((i & 1) == 0 && (j & 1) == 0) ||
					((i & 1) == 1 && (j & 1) == 1))
					outputImg.data.get()[w * i + j] = c[1];
				else if ((i & 1) == 0 && (j & 1) == 1)
					outputImg.data.get()[w * i + j] = c[2];
				else
					outputImg.data.get()[w * i + j] = c[0];
			}
		}
	} else if (bytePerChannel == 2) {
		unsigned short *data = reinterpret_cast<unsigned short *>(inputImg.data.get());
		unsigned short *odata = reinterpret_cast<unsigned short *>(outputImg.data.get());
		for (unsigned i = 0; i < h; i++) {
			for (unsigned j = 0; j < w; j++) {
				unsigned short *c = &data[w * i * 3 + j * 3];

				if (((i & 1) == 0 && (j & 1) == 0) ||
					((i & 1) == 1 && (j & 1) == 1))
					odata[w * i + j] = c[1];
				else if ((i & 1) == 0 && (j & 1) == 1)
					odata[w * i + j] = c[2];
				else
					odata[w * i + j] = c[0];
			}
		}
	}
	return FAST_OK;
}

fastStatus_t MosaicGRBG(Image<FastAllocator> &inputImg, Image<FastAllocator> &outputImg) {
	outputImg.w = inputImg.w;
	outputImg.h = inputImg.h;
	outputImg.wPitch = inputImg.w;
	outputImg.surfaceFmt = IdentifySurface(GetBitsPerChannelFromSurface(inputImg.surfaceFmt), 1);
	outputImg.samplingFmt = FAST_JPEG_Y;

	const unsigned bytePerChannel = GetBytesPerChannelFromSurface(inputImg.surfaceFmt);

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(outputImg.data.reset((unsigned char *)alloc.allocate(inputImg.w*inputImg.h*bytePerChannel)));

	const unsigned w = inputImg.w;
	const unsigned h = inputImg.h;
	if (bytePerChannel == 1) {
		for (unsigned i = 0; i < h; i++) {
			for (unsigned j = 0; j < w; j++) {
				unsigned char *c = &inputImg.data.get()[w * i * 3 + j * 3];
				if (((i & 1) == 0 && (j & 1) == 0) ||
					((i & 1) == 1 && (j & 1) == 1))
					outputImg.data.get()[w * i + j] = c[1];
				else if ((i & 1) == 0 && (j & 1) == 1)
					outputImg.data.get()[w * i + j] = c[0];
				else
					outputImg.data.get()[w * i + j] = c[2];
			}
		}
	} else if (bytePerChannel == 2) {
		unsigned short *data = reinterpret_cast<unsigned short *>(inputImg.data.get());
		unsigned short *odata = reinterpret_cast<unsigned short *>(outputImg.data.get());
		for (unsigned i = 0; i < h; i++) {
			for (unsigned j = 0; j < w; j++) {
				unsigned short *c = &data[w * i * 3 + j * 3];
				if (((i & 1) == 0 && (j & 1) == 0) ||
					((i & 1) == 1 && (j & 1) == 1))
					odata[w * i + j] = c[1];
				else if ((i & 1) == 0 && (j & 1) == 1)
					odata[w * i + j] = c[0];
				else
					odata[w * i + j] = c[2];
			}
		}
	}
	return FAST_OK;
}

fastStatus_t MosaicBGGR(Image<FastAllocator> &inputImg, Image<FastAllocator> &outputImg) {
	outputImg.w = inputImg.w;
	outputImg.h = inputImg.h;
	outputImg.wPitch = inputImg.w;
	outputImg.surfaceFmt = IdentifySurface(GetBitsPerChannelFromSurface(inputImg.surfaceFmt), 1);
	outputImg.samplingFmt = FAST_JPEG_Y;

	const unsigned bytePerChannel = GetBytesPerChannelFromSurface(inputImg.surfaceFmt);

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(outputImg.data.reset((unsigned char *)alloc.allocate(inputImg.w*inputImg.h*bytePerChannel)));

	const unsigned w = inputImg.w;
	const unsigned h = inputImg.h;
	if (bytePerChannel == 1) {
		for (unsigned i = 0; i < h; i++) {
			for (unsigned j = 0; j < w; j++) {
				unsigned char *c = &inputImg.data.get()[w * i * 3 + j * 3];
				if ((i & 1) == 0 && (j & 1) == 0)
					outputImg.data.get()[w * i + j] = c[2];
				else if ((i & 1) == 1 && (j & 1) == 1)
					outputImg.data.get()[w * i + j] = c[0];
				else
					outputImg.data.get()[w * i + j] = c[1];
			}
		}
	} else if (bytePerChannel == 2) {
		unsigned short *data = reinterpret_cast<unsigned short *>(inputImg.data.get());
		unsigned short *odata = reinterpret_cast<unsigned short *>(outputImg.data.get());
		for (unsigned i = 0; i < h; i++) {
			for (unsigned j = 0; j < w; j++) {
				unsigned short *c = &data[w * i * 3 + j * 3];
				if ((i & 1) == 0 && (j & 1) == 0)
					odata[w * i + j] = c[2];
				else if ((i & 1) == 1 && (j & 1) == 1)
					odata[w * i + j] = c[0];
				else
					odata[w * i + j] = c[1];
			}
		}
	}

	return FAST_OK;
}

fastStatus_t RunMosaic(int argc, char* argv[]) {
	Image<FastAllocator> inputImg;
	Image<FastAllocator> outputImage;

	char *ifile, *ofile;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "i", &ifile);
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "o", &ofile);

	printf("Input image: %s\n", ifile);
	printf("Output image: %s\n", ofile);

	CHECK_FAST(fvLoadImage(ifile, "", inputImg, 0, 0, 0, false));

	fastStatus_t ret;
	char *pattern = NULL;
	if (ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "pattern")) {
		ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "pattern", &pattern);

		if (strcmp(pattern, "RGGB") == 0)
			ret = MosaicRGGB(inputImg, outputImage);
		else if (strcmp(pattern, "BGGR") == 0)
			ret = MosaicBGGR(inputImg, outputImage);
		else if (strcmp(pattern, "GBRG") == 0)
			ret = MosaicGBRG(inputImg, outputImage);
		else if (strcmp(pattern, "GRBG") == 0)
			ret = MosaicGRBG(inputImg, outputImage);
		else {
			fprintf(stderr, "Pattern %s was not recognized.", pattern);
			ret = FAST_INVALID_VALUE;
		}
	} else ret = MosaicRGGB(inputImg, outputImage);

	if (ret == FAST_OK) {
		const unsigned pitch = outputImage.w * GetBytesPerChannelFromSurface(outputImage.surfaceFmt);
		return fvSaveImageToFile(ofile, outputImage.data, outputImage.surfaceFmt, outputImage.h, outputImage.w, pitch, false);
	}

	return ret;
}