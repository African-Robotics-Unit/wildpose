/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include <list>

#include "Decoder.h"

#include "JpegDecoderSampleOptions.h"

#include "Image.h"
#include "checks.h"
#include "supported_files.hpp"
#include "FastAllocator.h"
#include "helper_bytestream.hpp"
#include "DecoderCpu.h"
#include "helper_jpeg.hpp"
#include "BayerImageResize.h"

fastStatus_t RunJpegDecoder(JpegDecoderSampleOptions &options) {
	std::list< Bytestream<FastAllocator> > inputImgs;
	std::list< Image<FastAllocator> > outputImgs;

	options.SurfaceFmt = BaseOptions::GetSurfaceFormatFromExtension(options.OutputPath);

	if (options.IsFolder) {
		CHECK_FAST(fvLoadBytestreams(options.InputPath, inputImgs, false));
		int idx = 0;
		for (auto i = inputImgs.begin(); i != inputImgs.end(); ++i, idx++) {
			i->outputFileName = generateOutputFileName(options.OutputPath, idx);
		}
	} else {
		Bytestream< FastAllocator > inputImg;
		CHECK_FAST(fvLoadBytestream(std::string(options.InputPath), inputImg, false));
		inputImgs.push_back(inputImg);
		(--inputImgs.end())->outputFileName = std::string(options.OutputPath);
	}

	fastJfifInfo_t jfifInfo = { };
	CHECK_FAST(fastJfifHeaderLoadFromMemory((*inputImgs.begin()).data.get(), (*inputImgs.begin()).size, &jfifInfo));
	for (unsigned i = 0; i < jfifInfo.exifSectionsCount; i++) {
		free(jfifInfo.exifSections[i].exifData);
	}
	if (jfifInfo.exifSections != NULL) {
		free(jfifInfo.exifSections);
	}

	options.MaxHeight = options.MaxHeight == 0 ? jfifInfo.height : options.MaxHeight;
	options.MaxWidth = options.MaxWidth == 0 ? jfifInfo.width : options.MaxWidth;
	options.SurfaceFmt = IdentifySurface(jfifInfo.bitsPerChannel, jfifInfo.jpegFmt == FAST_JPEG_Y ? 1 : 3);

	Decoder hDecoder(options.Info, false);
	DecoderCpu hDecoderCpu(options.Info);

	if (options.SurfaceFmt == FAST_I12 || options.SurfaceFmt == FAST_RGB12) {
		CHECK_FAST(hDecoderCpu.Init(options));
		CHECK_FAST(hDecoderCpu.Decode(inputImgs, outputImgs));
	} else {
		CHECK_FAST(hDecoder.Init(options, NULL));
		CHECK_FAST(hDecoder.Decode(inputImgs, outputImgs, 0, NULL));
	}

	for (auto i = outputImgs.begin(); i != outputImgs.end(); ++i) {
		if (options.JpegDecoder.BayerCompression) {
			BayerSplitLines(*i);
		}
		if (!options.Discard) {
			CHECK_FAST(fvSaveImageToFile((char*)(*i).outputFileName.c_str(), (*i).data, (*i).surfaceFmt, (*i).bitsPerChannel, (*i).h, (*i).w, (*i).wPitch, false));
		}
	}

	inputImgs.clear();
	outputImgs.clear();

	if (options.SurfaceFmt == FAST_I12 || options.SurfaceFmt == FAST_RGB12) {
		CHECK_FAST(hDecoderCpu.Close());
	} else {
		CHECK_FAST(hDecoder.Close());
	}

	return FAST_OK;
}
