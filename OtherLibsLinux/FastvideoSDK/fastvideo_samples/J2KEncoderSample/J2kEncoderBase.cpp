/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "J2kEncoderBase.h"

#include <cstdio>

#include "fastvideo_sdk.h"
#include "supported_files.hpp"
#include "checks.h"

J2kEncoderBase::J2kEncoderBase(void) { };
J2kEncoderBase::~J2kEncoderBase(void) { };

fastStatus_t J2kEncoderBase::Init(J2kEncoderOptions &options) {
	if (options.SurfaceFmt == FAST_BGR8)
		options.SurfaceFmt = FAST_RGB8;

	this->options = options;
	folder = options.IsFolder;
	maxWidth = options.MaxWidth;
	maxHeight = options.MaxHeight;
	batchSize = options.BatchSize;
	surfaceFmt = options.SurfaceFmt;

	parameters = { 0 };
	{
		parameters.lossless = options.Algorithm == FAST_ENCODER_J2K_ALGORITHM_ENCODE_REVERSIBLE;
		parameters.pcrdEnabled = options.CompressionRatio > 1;
		parameters.dwtLevels = options.DWT_Levels;
		parameters.codeblockSize = options.CodeblockSize;
		parameters.maxQuality = options.Quality;
		parameters.compressionRatio = options.CompressionRatio;
		parameters.info = options.Info;
		parameters.tier2Threads = options.Tier2Threads;

		parameters.tileWidth = options.TileWidth;
		parameters.tileHeight = options.TileHeight;
		parameters.noMCT = options.NoMCT;

		if (parameters.codeblockSize == 0) parameters.codeblockSize = 32;

		parameters.ss1_x = options.ss1_x;
		parameters.ss1_y = options.ss1_y;

		parameters.ss2_x = options.ss2_x;
		parameters.ss2_y = options.ss2_y;

		parameters.ss3_x = options.ss3_x;
		parameters.ss3_y = options.ss3_y;
		parameters.yuvSubsampledFormat = false;
		parameters.outputBitDepth = options.OutputBitDepth;
		parameters.overwriteSurfaceBitDepth = options.OverwriteSurfaceBitDepth;
	}

	if (options.Info) {
		// Print active values of the main parameters.
		if (parameters.lossless)
			printf("Reversible (lossless) compression\n");
		else
			printf("Irreversible (lossy) compression\n");
		printf("%d Resolution levels\n", parameters.dwtLevels + 1);
		if (parameters.lossless)
			printf("Maximum quality (quantization disabled)\n");
		else {
			if (parameters.maxQuality >= 0.01f)
				printf("%.1f%% Quality\n", parameters.maxQuality * 100.0f);
			else
				printf("%g%% Quality\n", parameters.maxQuality * 100.0f);
			if (parameters.compressionRatio > 0)
				printf("%g:1 Compression ratio\n", parameters.compressionRatio);
			printf("%dx%d Codeblock size\n", parameters.codeblockSize, parameters.codeblockSize);
		}
	}

	return FAST_OK;
}
