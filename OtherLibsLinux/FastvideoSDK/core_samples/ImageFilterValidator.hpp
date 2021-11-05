/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __IMAGE_FILTER_VALIDATOR__
#define __IMAGE_FILTER_VALIDATOR__

#include "fastvideo_sdk.h"

#include <string>

typedef struct {
	fastImageFilterType_t imageFilterType;

	fastSurfaceFormat_t surfaces[3];
	std::string errorMessage;
} ImageFilterSurfaces_t;

#define FILTERS_COUNT 16
static const ImageFilterSurfaces_t ImageFilterSurfaces[FILTERS_COUNT] =
{
	{
		FAST_LUT_8_8,
		{
			FAST_I8,
			FAST_RGB8,
			FAST_BGR8
		},
		"Incorrect surface format (just RGB and Grayscale 8-bit)"
	},

	{
		FAST_LUT_8_8_C,
		{
			FAST_RGB8,
			FAST_BGR8,
			FAST_RGB8
		},
		"Incorrect surface format (just RGB 8-bit)"
	},

	{
		FAST_LUT_12_8,
		{
			FAST_RGB12,
			FAST_I12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB and Grayscale 12-bit)"
	},

	{
		FAST_LUT_12_8_C,
		{
			FAST_RGB12,
			FAST_RGB12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB 12-bit)"
	},

	{
		FAST_LUT_12_12,
		{
			FAST_RGB12,
			FAST_I12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB and Grayscale 12-bit)"
	},

	{
		FAST_LUT_12_12_C,
		{
			FAST_RGB12,
			FAST_RGB12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB 12-bit)"
	},

	{
		FAST_LUT_12_16,
		{
			FAST_RGB12,
			FAST_I12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB and Grayscale 12-bit)"
	},

	{
		FAST_LUT_12_16_C,
		{
			FAST_RGB12,
			FAST_RGB12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB 12-bit)"
	},

	{
		FAST_LUT_16_16,
		{
			FAST_RGB16,
			FAST_I16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB and Grayscale 16-bit)"
	},

	{
		FAST_LUT_16_16_C,
		{
			FAST_RGB16,
			FAST_RGB16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB 16-bit)"
	},

	{
		FAST_LUT_16_16_FR,
		{
			FAST_RGB16,
			FAST_I16,
			FAST_RGB16
		},
			"Incorrect surface format (just RGB and Grayscale 16-bit)"
	},

	{
		FAST_LUT_16_16_FR_C,
		{
			FAST_RGB16,
			FAST_RGB16,
			FAST_RGB16
		},
			"Incorrect surface format (just RGB 16-bit)"
	},

	{
		FAST_LUT_16_8,
		{
			FAST_RGB16,
			FAST_I16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB and Grayscale 16-bit)"
	},

	{
		FAST_LUT_16_8_C,
		{
			FAST_RGB16,
			FAST_RGB16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB 16-bit)"
	},

	{
		FAST_HSV_LUT_3D,
		{
			FAST_RGB12,
			FAST_RGB16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB 12/16-bit)"
	},

	{
		FAST_RGB_LUT_3D,
		{
			FAST_RGB12,
			FAST_RGB16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB 12/16-bit)"
	}
};

static bool ValidateSurface(fastImageFilterType_t imageFilter, fastSurfaceFormat_t surfaceFmt) {
	for (int i = 0; i < FILTERS_COUNT; i++) {
		if (ImageFilterSurfaces[i].imageFilterType == imageFilter) {
			for (int j = 0; j < 3; j++) {
				if (ImageFilterSurfaces[i].surfaces[j] == surfaceFmt) {
					return true;
				}
			}
			
			fprintf(stderr, "%s\n", ImageFilterSurfaces[i].errorMessage.c_str());
			return false;
		}
	}

	return false;
}

#endif // __IMAGE_FILTER_VALIDATOR__