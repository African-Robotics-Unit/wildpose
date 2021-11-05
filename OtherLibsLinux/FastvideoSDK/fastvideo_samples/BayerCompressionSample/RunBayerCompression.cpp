/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "RunBayerCompression.hpp"

#include "SplitterEncoder.h"
#include "DecoderMerger.h"

#include "checks.h"
#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "FastAllocator.h"
#include "BayerExifInfo.hpp"

#include "helper_bytestream.hpp"
#include "helper_jpeg.hpp"
#include "helper_quant_table.hpp"

fastStatus_t _DecodeJpeg(unsigned char *data, unsigned dataSize, fastJfifInfo_t *jfifInfo) {
	return fastJfifLoadFromMemory(
		data,
		dataSize,

		jfifInfo
	);
}

fastStatus_t RunSplitterEncoder(DebayerJpegSampleOptions options) {
	std::list< Image<FastAllocator> > inputImgs;

	if (options.IsFolder) {
		CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImgs, options.RawWidth, options.RawHeight, options.BitsPerChannel, false));
	} else {
		Image<FastAllocator> img;
		CHECK_FAST(fvLoadImage(std::string(options.InputPath), std::string(options.OutputPath), img, options.RawHeight, options.RawWidth, options.BitsPerChannel, false));
		options.MaxHeight = options.MaxHeight == 0 ? img.h : options.MaxHeight;
		options.MaxWidth = options.MaxWidth == 0 ? img.w : options.MaxWidth;
		inputImgs.push_back(img);
	}

	options.SurfaceFmt = (*inputImgs.begin()).surfaceFmt;
	printf("Output surface format: %s\n", EnumToString(options.SurfaceFmt));

	fastJpegQuantState_t quantState = { 0 };
	bool quantStateEnabled = false;
	if (options.JpegEncoder.QuantTableFileName != NULL) {
		printf("External quant table: %s\n", options.JpegEncoder.QuantTableFileName);
		CHECK_FAST(fvLoadQuantTable(options.JpegEncoder.QuantTableFileName, quantState));
		quantStateEnabled = true;
	}

	SplitterEncoder hSplitterEncoder(options.Info);
	CHECK_FAST(hSplitterEncoder.Init(options));
	CHECK_FAST(hSplitterEncoder.Transform(inputImgs, quantStateEnabled ? &quantState : NULL));

	inputImgs.clear();

	CHECK_FAST(hSplitterEncoder.Close());

	return FAST_OK;
}

fastStatus_t RunDecoderMerger(DebayerJpegSampleOptions options) {
	unsigned maxRestoredWidth = 0, maxRestoredHeight = 0;

	std::list<Bytestream<FastAllocator> > inputImgs;

	if (options.IsFolder) {
		CHECK_FAST(fvLoadBytestreams(options.InputPath, inputImgs, false));
		int idx = 0;
		for (auto i = inputImgs.begin(); i != inputImgs.end(); i++, idx++) {
			i->outputFileName = generateOutputFileName(options.OutputPath, idx);
		}
	} else {
		Bytestream< FastAllocator >  inputImg;
		CHECK_FAST(fvLoadBytestream(std::string(options.InputPath), inputImg, false));
		inputImgs.push_back(inputImg);

		(--inputImgs.end())->outputFileName = std::string(options.OutputPath);
	}

	fastJfifInfo_t jfifInfo = { };
	jfifInfo.h_Bytestream = NULL;
	jfifInfo.exifSections = NULL;
	jfifInfo.exifSectionsCount = 0;

	jfifInfo.bytestreamSize = (*inputImgs.begin()).size;
	CHECK_FAST(fastMalloc((void **)&jfifInfo.h_Bytestream, jfifInfo.bytestreamSize));

	CHECK_FAST(_DecodeJpeg((*inputImgs.begin()).data.get(), (*inputImgs.begin()).size, &jfifInfo));

	if (jfifInfo.h_Bytestream != NULL)
		CHECK_FAST_DEALLOCATION(fastFree(jfifInfo.h_Bytestream));

	if (jfifInfo.exifSectionsCount < 1) {
		fprintf(stderr, "Incorrect JPEG: EXIF sections was not found\n");
		return FAST_IO_ERROR;
	}

	if (jfifInfo.exifSections != NULL) {
		for (unsigned i = 0; i < jfifInfo.exifSectionsCount; i++) {
			ParseSplitterExif(&jfifInfo.exifSections[i], options.Debayer.BayerFormat, maxRestoredWidth, maxRestoredHeight);

			free(jfifInfo.exifSections[i].exifData);
		}

		free(jfifInfo.exifSections);
	}

	if (maxRestoredWidth == 0 || maxRestoredHeight == 0) {
		fprintf(stderr, "Incorrect JPEG: debayer parameters in EXIF section was not found\n");
		return FAST_IO_ERROR;
	}

	if (jfifInfo.jpegFmt != FAST_JPEG_Y) {
		fprintf(stderr, "Incorrect JPEG: just gray images supported\n");
		return FAST_IO_ERROR;
	}

	options.MaxHeight = options.MaxHeight == 0 ? jfifInfo.height : options.MaxHeight;
	options.MaxWidth = options.MaxWidth == 0 ? jfifInfo.width : options.MaxWidth;
	options.SurfaceFmt = IdentifySurface(jfifInfo.bitsPerChannel, 1);

	fastSurfaceFormat_t outputFmt = BaseOptions::GetSurfaceFormatFromExtension(options.OutputPath);
	if (jfifInfo.bitsPerChannel == 12) {
		outputFmt = FAST_I12;
	} else if (jfifInfo.bitsPerChannel == 16) {
		outputFmt = FAST_I16;
	}

	printf("Output surface format: %s\n", EnumToString(outputFmt));

	DecoderMerger hDecoderMerger(options.Info);
	CHECK_FAST(hDecoderMerger.Init(options, maxRestoredWidth, maxRestoredHeight));
	CHECK_FAST(hDecoderMerger.Transform(inputImgs));

	inputImgs.clear();

	CHECK_FAST(hDecoderMerger.Close());

	return FAST_OK;
}
