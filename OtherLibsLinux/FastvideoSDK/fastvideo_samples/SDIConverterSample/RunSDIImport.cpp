/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "RunSDIConverter.hpp"

#include "SDIImportFromHost.h"
#include "SDIImportFromDevice.h"

#include "checks.h"
#include "SDICommon.hpp"

#include "SurfaceTraits.hpp"
#include "EnumToStringSdk.h"
#include "HelperSDI.hpp"

fastStatus_t RunSDIImport(SDIConverterSampleOptions &options) {
	Image<FastAllocator> img;
	
	img.w = options.SDI.Width;
	img.h = options.SDI.Height;

	if (options.MaxWidth == 0) {
		options.MaxWidth = img.w;
	}
	if (options.MaxHeight == 0) {
		options.MaxHeight = img.h;
	}

	printf("Input image: %s\nImage size: %dx%d pixels\n\n", options.InputPath, img.w, img.h);
	printf("SDI format: %s\n", EnumToString(options.SDI.SDIFormat));

	CHECK_FAST(fvLoadBinary(options.InputPath, img.data));

	options.SurfaceFmt = GetSDISurfaceFormat(options.SDI.SDIFormat);
	options.BitsPerChannel = GetBitsPerChannelFromSurface(options.SurfaceFmt);

	if (options.SDI.IsGpu) {
		SDIImportFromDevice hSDIImport(options.Info);
		CHECK_FAST(hSDIImport.Init(options));
		CHECK_FAST(hSDIImport.Transform(img, options.OutputPath));

		if (IsSDICopy3Format(options.SDI.SDIFormat) && options.SDI.FileNameAlternate != NULL) {
			CHECK_FAST(hSDIImport.Transform3(img, options.SDI.FileNameAlternate));
		}

		CHECK_FAST(hSDIImport.Close());
	} else {
		SDIImportFromHost hSDIImport(options.Info);
		CHECK_FAST(hSDIImport.Init(options));
		CHECK_FAST(hSDIImport.Transform(img, options.OutputPath));

		if (IsSDICopy3Format(options.SDI.SDIFormat) && options.SDI.FileNameAlternate != NULL) {
			CHECK_FAST(hSDIImport.Transform3(img, options.SDI.FileNameAlternate));
		}

		CHECK_FAST(hSDIImport.Close());
	}

	return FAST_OK;
}
