/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "RunLuts.hpp"

#include "Lut8.h"
#include "Lut8C.h"
#include "Lut16.h"
#include "LutBayer.h"
#include "HSVLut3D.h"
#include "RGBLut3D.h"

#include "Image.h"
#include "FastAllocator.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "helper_lut.hpp"
#include "helper_lut_3d.hpp"

#include "ImageFilterValidator.hpp"
#include "SurfaceTraits.hpp"

fastStatus_t RunLut8c(LutSampleOptions &options) {
	std::list< Image<FastAllocator> > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, 8, false));
	} else {
		Image<FastAllocator> img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 8, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	for (auto i = inputImg.begin(); i != inputImg.end(); i++) {
		if ((*i).surfaceFmt == FAST_I8) {
			// validate input images
			fprintf(stderr, "Input file must be color\n");
			return FAST_IO_ERROR;
		}
	}

	printf("Input surface format: %s\n", EnumToString((*inputImg.begin()).surfaceFmt));
	printf("Output surface format: %s\n", EnumToString(options.SurfaceFmt));

	std::unique_ptr<unsigned char, FastAllocator> lutDataR, lutDataG, lutDataB;
	printf("\nLUT (R channel) file: %s\n", options.Lut.Lut_R);
	CHECK_FAST(fvLoadLut(options.Lut.Lut_R, lutDataR, 256));
	printf("LUT (G channel) file: %s\n", options.Lut.Lut_G);
	CHECK_FAST(fvLoadLut(options.Lut.Lut_G, lutDataG, 256));
	printf("LUT (B channel) file: %s\n", options.Lut.Lut_B);
	CHECK_FAST(fvLoadLut(options.Lut.Lut_B, lutDataB, 256));

	Lut8C hLut(options.Info);
	CHECK_FAST(hLut.Init(options, lutDataR.get(), lutDataG.get(), lutDataB.get()));
	CHECK_FAST(hLut.Transform(inputImg, lutDataR.get(), lutDataG.get(), lutDataB.get()));

	inputImg.clear();
	CHECK_FAST(hLut.Close());

	return FAST_OK;
}

fastStatus_t RunLut8(LutSampleOptions &options) {
	std::list< Image<FastAllocator> > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, 8, false));
	} else {
		Image<FastAllocator> img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 8, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	for (auto i = inputImg.begin(); i != inputImg.end(); i++) {
		if ((*i).surfaceFmt != FAST_I8) {
			// validate input images
			fprintf(stderr, "Input file must be gray (8 bit)\n");
			return FAST_IO_ERROR;
		}
	}

	printf("Input surface format: %s\n", EnumToString((*inputImg.begin()).surfaceFmt));
	printf("Output surface format: %s\n", EnumToString(options.SurfaceFmt));

	std::unique_ptr<unsigned char, FastAllocator> lutData;
	printf("\nLUT file: %s\n", options.Lut.Lut);
	CHECK_FAST(fvLoadLut(options.Lut.Lut, lutData, 256));

	Lut8 hLut(options.Info);
	CHECK_FAST(hLut.Init(options, lutData.get()));
	CHECK_FAST(hLut.Transform(inputImg, lutData.get()));

	inputImg.clear();
	CHECK_FAST(hLut.Close());

	return FAST_OK;
}

fastStatus_t RunLut16(LutSampleOptions &options) {
	std::list< Image<FastAllocator> > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, 12, false));
	} else {
		Image<FastAllocator> img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 12, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	for (auto i = inputImg.begin(); i != inputImg.end(); i++) {
		if (!ValidateSurface(options.Lut.ImageFilter, (*i).surfaceFmt)) {
			return FAST_IO_ERROR;
		}
	}
	options.SurfaceFmt = (*inputImg.begin()).surfaceFmt;

	printf("Input surface format: %s\n", EnumToString(options.SurfaceFmt));

	unsigned bitsPerPixel = GetBitsPerChannelFromSurface(options.SurfaceFmt);
	if (bitsPerPixel == 16 &&
		options.Lut.ImageFilter != FAST_LUT_16_16_FR &&
		options.Lut.ImageFilter != FAST_LUT_16_16_FR_C &&
		options.Lut.ImageFilter != FAST_LUT_16_16_FR_BAYER) {
		bitsPerPixel = 14;
	}

	void *lutR = NULL, *lutG = NULL, *lutB = NULL;
	std::unique_ptr<unsigned short, FastAllocator> lutDataR_16, lutDataG_16, lutDataB_16;
	std::unique_ptr<unsigned char, FastAllocator> lutDataR_8, lutDataG_8, lutDataB_8;

	if (options.Lut.ImageFilter == FAST_LUT_12_8 || options.Lut.ImageFilter == FAST_LUT_12_8_C ||
		options.Lut.ImageFilter == FAST_LUT_16_8 || options.Lut.ImageFilter == FAST_LUT_16_8_C) {
		if (options.Lut.Lut != NULL) {
			printf("\nLUT file: %s\n", options.Lut.Lut);
			CHECK_FAST(fvLoadLut(options.Lut.Lut, lutDataR_8, 1 << bitsPerPixel));

			lutR = lutDataR_8.get();
		} else {
			printf("\nLUT (R channel) file: %s\n", options.Lut.Lut_R);
			CHECK_FAST(fvLoadLut(options.Lut.Lut_R, lutDataR_8, 1 << bitsPerPixel));
			printf("LUT (G channel) file: %s\n", options.Lut.Lut_R);
			CHECK_FAST(fvLoadLut(options.Lut.Lut_G, lutDataG_8, 1 << bitsPerPixel));
			printf("LUT (B channel) file: %s\n", options.Lut.Lut_R);
			CHECK_FAST(fvLoadLut(options.Lut.Lut_B, lutDataB_8, 1 << bitsPerPixel));

			lutR = lutDataR_8.get();
			lutG = lutDataG_8.get();
			lutB = lutDataB_8.get();
		}
	} else {
		if (options.Lut.Lut != NULL) {
			printf("\nLUT file: %s\n", options.Lut.Lut);
			CHECK_FAST(fvLoadLut(options.Lut.Lut, lutDataR_16, 1 << bitsPerPixel));

			lutR = lutDataR_16.get();
		} else {
			printf("\nLUT (R channel) file: %s\n", options.Lut.Lut_R);
			CHECK_FAST(fvLoadLut(options.Lut.Lut_R, lutDataR_16, 1 << bitsPerPixel));
			printf("LUT (G channel) file: %s\n", options.Lut.Lut_R);
			CHECK_FAST(fvLoadLut(options.Lut.Lut_G, lutDataG_16, 1 << bitsPerPixel));
			printf("LUT (B channel) file: %s\n", options.Lut.Lut_R);
			CHECK_FAST(fvLoadLut(options.Lut.Lut_B, lutDataB_16, 1 << bitsPerPixel));

			lutR = lutDataR_16.get();
			lutG = lutDataG_16.get();
			lutB = lutDataB_16.get();
		}
	}

	Lut16 hLut(options.Info);
	CHECK_FAST(hLut.Init(options, lutR, lutG, lutB));
	CHECK_FAST(hLut.Transform(inputImg, lutR, lutG, lutB));

	inputImg.clear();
	CHECK_FAST(hLut.Close());

	return FAST_OK;
}

fastStatus_t RunLutBayer(LutDebayerSampleOptions &options) {
	std::list< Image<FastAllocator> > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, 8, false));
	} else {
		Image<FastAllocator > img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 8, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	for (auto i = inputImg.begin(); i != inputImg.end(); ++i) {
		if (GetNumberOfChannelsFromSurface((*i).surfaceFmt) != 1) {
			// validate input images
			fprintf(stderr, "Input file must be gray-scale\n");
			return FAST_IO_ERROR;
		}
	}
	options.SurfaceFmt = (inputImg.begin())->surfaceFmt;

	printf("Input surface format: %s\n", EnumToString((*inputImg.begin()).surfaceFmt));
	printf("Output surface format: %s\n", EnumToString(options.SurfaceFmt));
	printf("Debayer pattern: %s\n", EnumToString(options.Debayer.BayerFormat));

	std::unique_ptr<unsigned short, FastAllocator> lutDataR, lutDataG, lutDataB;
	{
		unsigned lutSize = 0;
		switch (options.Lut.ImageFilter) {
			case FAST_LUT_8_16_BAYER:
				lutSize = 256;
				break;
			case FAST_LUT_10_16_BAYER:
				lutSize = 1024;
				options.SurfaceFmt = FAST_I10;
				break;
			case FAST_LUT_12_16_BAYER:
				lutSize = 4096;
				options.SurfaceFmt = FAST_I12;
				break;
			case FAST_LUT_14_16_BAYER:
			case FAST_LUT_16_16_BAYER:
				lutSize = 16384;
				options.SurfaceFmt = FAST_I14;
				break;
			case FAST_LUT_16_16_FR_BAYER:
				lutSize = 65536;
				break;
			default:
				printf("Unsupported LUT format\n");
				return FAST_INVALID_VALUE;
		}
		printf("\nLUT (R channel) file: %s\n", options.Lut.Lut_R);
		CHECK_FAST(fvLoadLut(options.Lut.Lut_R, lutDataR, lutSize));
		printf("LUT (G channel) file: %s\n", options.Lut.Lut_G);
		CHECK_FAST(fvLoadLut(options.Lut.Lut_G, lutDataG, lutSize));
		printf("LUT (B channel) file: %s\n", options.Lut.Lut_B);
		CHECK_FAST(fvLoadLut(options.Lut.Lut_B, lutDataB, lutSize));
	}

	LutBayer hLut(options.Info);
	CHECK_FAST(hLut.Init(options, lutDataR.get(), lutDataG.get(), lutDataB.get()));
	CHECK_FAST(hLut.Transform(inputImg, NULL, NULL, NULL));

	inputImg.clear();
	CHECK_FAST(hLut.Close());

	return FAST_OK;
}

fastStatus_t RunLutHsv3D(LutSampleOptions &options) {
	if (options.Lut.Lut == NULL) {
		fprintf(stderr, "Input LUT file was not set\n");
		return FAST_IO_ERROR;
	}

	std::list< Image<FastAllocator > > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, 12, false));
	} else {
		Image<FastAllocator > img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 12, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	for (auto i = inputImg.begin(); i != inputImg.end(); i++) {
		if (!ValidateSurface(options.Lut.ImageFilter, (*i).surfaceFmt)) {
			return FAST_IO_ERROR;
		}
	}
	options.SurfaceFmt = (*inputImg.begin()).surfaceFmt;

	printf("Input surface format: %s\n", EnumToString(options.SurfaceFmt));
	printf("\nLUT file: %s\n", options.Lut.Lut);

	std::unique_ptr<float, FastAllocator> lutData_h;
	std::unique_ptr<float, FastAllocator> lutData_s;
	std::unique_ptr<float, FastAllocator> lutData_v;
	CHECK_FAST(fvLoadXml3D(options.Lut.Lut, lutData_h, lutData_s, lutData_v, &options.Lut.lutSize3D, options.Lut.OperationType));

	HsvLut3D hLut(options.Info);
	CHECK_FAST(hLut.Init(options, lutData_h.get(), lutData_s.get(), lutData_v.get(), options.Lut.lutSize3D));
	CHECK_FAST(hLut.Transform(inputImg, lutData_h.get(), lutData_s.get(), lutData_v.get(), options.Lut.lutSize3D));

	inputImg.clear();
	CHECK_FAST(hLut.Close());

	return FAST_OK;
}

fastStatus_t RunLutRgb3D(LutSampleOptions &options) {
	if (options.Lut.Lut == NULL) {
		printf("Input LUT file was not set. Bypass mode enabled\n");
		if (options.Lut.lutSize3D.x == 0 || options.Lut.lutSize3D.y == 0 || options.Lut.lutSize3D.z == 0) {
			printf("LUT size was not set\n");
			return FAST_INVALID_SIZE;
		}
	}

	std::list<Image<FastAllocator > > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, 12, false));
	} else {
		Image<FastAllocator > img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 12, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	for (auto i = inputImg.begin(); i != inputImg.end(); i++) {
		if (!ValidateSurface(options.Lut.ImageFilter, (*i).surfaceFmt)) {
			return FAST_IO_ERROR;
		}
	}
	options.SurfaceFmt = (*inputImg.begin()).surfaceFmt;

	printf("Input surface format: %s\n", EnumToString(options.SurfaceFmt));
	printf("\nLUT file: %s\n", options.Lut.Lut);

	std::unique_ptr<float, FastAllocator> lutData_r;
	std::unique_ptr<float, FastAllocator> lutData_g;
	std::unique_ptr<float, FastAllocator> lutData_b;
	if (options.Lut.Lut != NULL) {
		CHECK_FAST(fvLoadCube3D(options.Lut.Lut, lutData_r, lutData_g, lutData_b, options.Lut.lutSize1D));
		if (options.Lut.lutSize3D.x == 0 || options.Lut.lutSize3D.y == 0 || options.Lut.lutSize3D.z == 0) {
			if (options.Lut.lutSize1D != 0) {
				options.Lut.lutSize3D.x = options.Lut.lutSize3D.y = options.Lut.lutSize3D.z = options.Lut.lutSize1D;
			}
		}
	}

	RgbLut3D hLut(options.Info);
	CHECK_FAST(hLut.Init(
		options,
		options.Lut.Lut != NULL ? lutData_r.get() : NULL,
		options.Lut.Lut != NULL ? lutData_g.get() : NULL,
		options.Lut.Lut != NULL ? lutData_b.get() : NULL,
		options.Lut.lutSize3D
	));
	CHECK_FAST(hLut.Transform(inputImg, NULL, NULL, NULL, options.Lut.lutSize3D));

	inputImg.clear();
	CHECK_FAST(hLut.Close());

	return FAST_OK;
}

fastStatus_t RunLutRgb3D_Bypass(LutSampleOptions &options) {
	if (options.Lut.Lut == NULL) {
		printf("Input LUT file was not set. Bypass mode enabled\n");
		if (options.Lut.lutSize3D.x == 0 || options.Lut.lutSize3D.y == 0 || options.Lut.lutSize3D.z == 0) {
			printf("LUT size was not set\n");
			return FAST_INVALID_SIZE;
		}
	}

	std::list< Image<FastAllocator > > inputImg;
	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, 0, 0, 12, false));
	} else {
		Image<FastAllocator > img;

		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 12, false));

		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImg.push_back(img);
	}

	for (auto i = inputImg.begin(); i != inputImg.end(); i++) {
		if (!ValidateSurface(options.Lut.ImageFilter, (*i).surfaceFmt)) {
			return FAST_IO_ERROR;
		}
	}
	options.SurfaceFmt = (*inputImg.begin()).surfaceFmt;

	printf("Input surface format: %s\n", EnumToString(options.SurfaceFmt));
	printf("\nLUT file: %s\n", options.Lut.Lut);

	std::unique_ptr<float, FastAllocator> lutData_r;
	std::unique_ptr<float, FastAllocator> lutData_g;
	std::unique_ptr<float, FastAllocator> lutData_b;
	if (options.Lut.Lut != NULL) {
		CHECK_FAST(fvLoadCube3D(options.Lut.Lut, lutData_r, lutData_g, lutData_b, options.Lut.lutSize1D));
		if (options.Lut.lutSize3D.x == 0 || options.Lut.lutSize3D.y == 0 || options.Lut.lutSize3D.z == 0) {
			if (options.Lut.lutSize1D != 0) {
				options.Lut.lutSize3D.x = options.Lut.lutSize3D.y = options.Lut.lutSize3D.z = options.Lut.lutSize1D;
			}
		}
	}

	RgbLut3D hLut(options.Info);
	CHECK_FAST(hLut.Init(
		options,
		options.Lut.Lut != NULL ? lutData_r.get() : NULL,
		options.Lut.Lut != NULL ? lutData_g.get() : NULL,
		options.Lut.Lut != NULL ? lutData_b.get() : NULL,
		options.Lut.lutSize3D
	));

	options.Lut.lutSize3D.x = options.Lut.lutSize3D.y = options.Lut.lutSize3D.z = 0;
	CHECK_FAST(hLut.Transform(
		inputImg,
		options.Lut.Lut != NULL ? lutData_r.get() : NULL,
		options.Lut.Lut != NULL ? lutData_g.get() : NULL,
		options.Lut.Lut != NULL ? lutData_b.get() : NULL,
		options.Lut.lutSize3D
	));

	inputImg.clear();
	CHECK_FAST(hLut.Close());

	return FAST_OK;
}