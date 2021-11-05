/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "supported_files.hpp"
#include "EnumToStringSdk.h"

#include "Image.h"
#include "BGRXImport.h"
#include "BaseOptions.h"

#include "FastAllocator.h"

fastStatus_t RunBGRXImport(BaseOptions &options) {
	Image<FastAllocator> img, xrgbImg;
	CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, 0, 0, 8, false));
	if (img.surfaceFmt != FAST_RGB8) {
		fprintf(stderr, "Input file must be 8-bit color\n");
		return FAST_IO_ERROR;
	}

	xrgbImg.surfaceFmt = FAST_BGRX8;
	xrgbImg.w = img.w;
	xrgbImg.h = img.h;
	xrgbImg.wPitch = GetPitchFromSurface(xrgbImg.surfaceFmt, xrgbImg.w);
	xrgbImg.bitsPerChannel = img.bitsPerChannel;
	xrgbImg.inputFileName = img.inputFileName;
	xrgbImg.outputFileName = img.outputFileName;

	options.MaxHeight = options.MaxHeight == 0 ? xrgbImg.h : options.MaxHeight;
	options.MaxWidth = options.MaxWidth == 0 ? xrgbImg.w : options.MaxWidth;
	options.SurfaceFmt = xrgbImg.surfaceFmt;
	options.BitsPerChannel = xrgbImg.bitsPerChannel;

	FastAllocator alloc;
	xrgbImg.data.reset((unsigned char *)alloc.allocate(xrgbImg.wPitch * xrgbImg.h));
	{
		unsigned char *src = img.data.get();
		unsigned char *dst = xrgbImg.data.get();
		for (unsigned i = 0; i < xrgbImg.h; i++) {
			for (unsigned j = 0; j < xrgbImg.w; j++) {
				dst[i *xrgbImg.wPitch + j * 4 + 2] = src[i * img.wPitch + j * 3 + 0];
				dst[i *xrgbImg.wPitch + j * 4 + 1] = src[i * img.wPitch + j * 3 + 1];
				dst[i *xrgbImg.wPitch + j * 4 + 0] = src[i * img.wPitch + j * 3 + 2];
			}
		}
	}
	
	printf("Input surface format: BGRX\n");
	printf("Output surface format: %s\n", EnumToString(BaseOptions::GetSurfaceFormatFromExtension(options.OutputPath)));

	BGRXImport hBGRXImport(options.Info);
	CHECK_FAST(hBGRXImport.Init(options));
	CHECK_FAST(hBGRXImport.Transform(xrgbImg));
	CHECK_FAST(hBGRXImport.Close());

	return FAST_OK;
}
