/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __RUN_RAW_IMPORT__
#define __RUN_RAW_IMPORT__

#include "supported_files.hpp"
#include "EnumToStringSdk.h"

#include "Image.h"

#include "RawImportFromHost.h"
#include "RawImportFromDevice.h"

#include "RawImportSampleOptions.h"

#include "FastAllocator.h"

int GetBitsPerChannel(const fastRawFormat_t rawFmt) {
	switch (rawFmt) {
		case FAST_RAW_XIMEA12:
			return 12;
		case FAST_RAW_PTG12:
			return 12;
		default:
			return 0;
	}
}

fastStatus_t RunRawImport(RawImportSampleOptions &options) {
	Image<FastAllocator> img;
	CHECK_FAST(fvLoadImage(
		std::string(options.InputPath),
		std::string(options.OutputPath),
		img, options.Raw.Height, options.Raw.Width,
		GetBitsPerChannel(options.Raw.RawFormat), false
	));
	if (img.surfaceFmt != FAST_I12) {
		fprintf(stderr, "Input file must be 12-bit gray\n");
		return FAST_IO_ERROR;
	}

	options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
	options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
	options.SurfaceFmt = img.surfaceFmt;
	options.BitsPerChannel = img.bitsPerChannel;

	printf("Input surface format: grayscale (%s)\n", EnumToString(options.Raw.RawFormat));
	printf("Output surface format: %s\n", EnumToString(BaseOptions::GetSurfaceFormatFromExtension(options.OutputPath)));
	printf("GPU mode: %s\n", options.Raw.IsGpu ? "true" : "false");

	if (options.Raw.IsGpu) {
		RawImportFromDevice hRawImport(options.Info);
		CHECK_FAST(hRawImport.Init(options));
		CHECK_FAST(hRawImport.Transform(img));
		CHECK_FAST(hRawImport.Close());
	} else {
		RawImportFromHost hRawImport(options.Info);
		CHECK_FAST(hRawImport.Init(options));
		CHECK_FAST(hRawImport.Transform(img));
		CHECK_FAST(hRawImport.Close());
	}

	return FAST_OK;
}

#endif // __RUN_RAW_IMPORT__