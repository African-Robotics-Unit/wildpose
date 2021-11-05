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

#include "SDIExportToHost.h"
#include "SDIExportToDevice.h"
#include "checks.h"
#include "SDICommon.hpp"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "HelperSDI.hpp"

fastStatus_t RunSDIExport(SDIConverterSampleOptions &options) {
	Image<FastAllocator> img;
	CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 8, false));

	options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
	options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;

	options.SurfaceFmt = img.surfaceFmt;
	options.BitsPerChannel = GetBitsPerChannelFromSurface(img.surfaceFmt);

	printf("Input surface format: %s\n", EnumToString(img.surfaceFmt));
	printf("Input image: %s\nImage size: %dx%d pixels\n\n", options.InputPath, img.w, img.h);
	printf("SDI format: %s\n", EnumToString(options.SDI.SDIFormat));

	if (options.SDI.IsGpu) {
		SDIExportToDevice hSDIExport(options.Info);
		CHECK_FAST(hSDIExport.Init(options));
		CHECK_FAST(hSDIExport.Transform(img, options.OutputPath));
		if (IsSDICopy3Format(options.SDI.SDIFormat) && options.SDI.FileNameAlternate != nullptr)
			CHECK_FAST(hSDIExport.Transform3(img, options.SDI.FileNameAlternate));
		CHECK_FAST(hSDIExport.Close());
	} else {
		SDIExportToHost hSDIExport(options.Info);
		CHECK_FAST(hSDIExport.Init(options));
		CHECK_FAST(hSDIExport.Transform(img, options.OutputPath));
		if (IsSDICopy3Format(options.SDI.SDIFormat) && options.SDI.FileNameAlternate != nullptr)
			CHECK_FAST(hSDIExport.Transform3(img, options.SDI.FileNameAlternate));
		CHECK_FAST(hSDIExport.Close());
	}

	return FAST_OK;
}
