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
#include "FastAllocator.h"

#include "checks.h"
#include "supported_files.hpp"
#include "helper_bytestream.hpp"

#include "J2kDecoderOptions.h"
#include "J2kDecoderOneImage.h"
#include "J2kDecoderBatch.h"

#include "fastvideo_decoder_j2k.h"

#include "J2kDecoderHelper.h"

fastStatus_t RunJ2kDecoder(J2kDecoderOptions options) {
	std::list<Bytestream<FastAllocator> > inputImgs;
	std::list<Image<FastAllocator> > outputImgs;

    {
		fastSdkParametersHandle_t hSdkParameters;
		CHECK_FAST(fastGetSdkParametersHandle(&hSdkParameters));
		CHECK_FAST(fastDecoderJ2kLibraryInit(hSdkParameters));
	}

	options.SurfaceFmt = BaseOptions::GetSurfaceFormatFromExtension(options.OutputPath);

	if (options.IsFolder) {
		CHECK_FAST(fvLoadBytestreams(options.InputPath, inputImgs, false));
	} else {
		Bytestream< FastAllocator > inputImg;
		CHECK_FAST(fvLoadBytestream(std::string(options.InputPath), inputImg, false));
		inputImgs.push_back(inputImg);
		(--inputImgs.end())->outputFileName = std::string(options.OutputPath);
	}

	fastJ2kImageInfo_t j2kInfo = { };

	CHECK_FAST(fastDecoderJ2kPredecode(
		&j2kInfo,
		inputImgs.begin()->data.get(),
		inputImgs.begin()->size
	));
	options.MaxWidth = std::max(options.MaxWidth, j2kInfo.width);
	options.MaxHeight = std::max(options.MaxHeight, j2kInfo.height);
    options.SurfaceFmt = options.ForceTo8bits ? IdentifySurface(8, GetNumberOfChannelsFromSurface(j2kInfo.decoderSurfaceFmt)) : j2kInfo.decoderSurfaceFmt;

    if (j2kInfo.tileWidth != j2kInfo.width || j2kInfo.tileHeight != j2kInfo.height) {
	options.MaxTileWidth = std::max<unsigned>(options.MaxTileWidth, j2kInfo.tileWidth);
	options.MaxTileHeight = std::max<unsigned>(options.MaxTileHeight, j2kInfo.tileHeight);
    }

    if (options.BitsPerChannel == 0) {
        for (int i = 0; i < j2kInfo.componentCount; i++)
            options.BitsPerChannel = std::max((int)options.BitsPerChannel, j2kInfo.components[i].bitDepth);
    }
	
	if (!options.IsEnabledWindow) {
		options.WindowLeftTopCoordsX = options.WindowLeftTopCoordsY = 0;
		options.WindowWidth = options.MaxWidth;
		options.WindowHeight = options.MaxHeight;
	}

	if (options.BatchSize > inputImgs.size() * options.RepeatCount)
		options.RepeatCount = options.BatchSize;

    if (options.PrintBoxes)
	{
        if (j2kInfo.containsRreqBox) PrintJ2kReaderRequirementBox(&j2kInfo);
        if (j2kInfo.asocBoxesCount > 0) PrintJ2kAsocBoxes(&j2kInfo);
        if (j2kInfo.uuidBoxesCount > 0 || j2kInfo.containsUuidInfoBox) PrintJ2kUUIDboxes(&j2kInfo);
    }
    if (options.PrintGML)
    {
        PrintJ2kGML(&j2kInfo);
    }
    if (options.OutputPathGML != nullptr)
    {
        WriteJ2kGML(options.OutputPathGML, &j2kInfo);
	}

	if (options.BatchSize == 1) {
		J2kDecoderOneImage decoder;
		CHECK_FAST(decoder.Init(options, &j2kInfo));
		CHECK_FAST(decoder.Transform(inputImgs, outputImgs));
		CHECK_FAST(decoder.Close());
	} else if (options.BatchSize > 1) {
		MtResult result;

		J2kDecoderBatch decoder(false);
		CHECK_FAST(decoder.Init(options, &j2kInfo, &result));
		CHECK_FAST(decoder.Transform(inputImgs, outputImgs, &result));
		CHECK_FAST(decoder.Close());
	}

	int idx = 0;
	for (auto i = outputImgs.begin(); i != outputImgs.end(); ++i) {
		std::string outputFileName = generateOutputFileName(options.OutputPath, idx);
        if ((*i).data == nullptr) continue;
		CHECK_FAST(fvSaveImageToFile((char *)outputFileName.c_str(), (*i).data, (*i).surfaceFmt, (*i).bitsPerChannel, (*i).h, (*i).w, (*i).wPitch, false));
		idx++;
	}

	inputImgs.clear();
	outputImgs.clear();

	FreeJ2kImageInfo(&j2kInfo);
	return FAST_OK;
}
