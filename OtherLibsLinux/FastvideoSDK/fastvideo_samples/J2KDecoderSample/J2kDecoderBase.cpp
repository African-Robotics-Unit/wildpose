/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "J2kDecoderBase.h"
#include "SurfaceTraits.hpp"
#include <cstring>

J2kDecoderBase::J2kDecoderBase() {
	parameters = { 0 };
}

fastStatus_t J2kDecoderBase::Init(J2kDecoderOptions &options, fastJ2kImageInfo_t *sampleImage) {
	this->options = options;

	// Select the appropriate surface format
	parameters.resolutionLevels = options.ResolutionLevels;
	parameters.verboseLevel = options.Info ? 1 : 0;
	parameters.enableROI = options.EnableROI;

	parameters.maxTileHeight = options.MaxTileHeight;
	parameters.maxTileWidth = options.MaxTileWidth;

	parameters.windowX0 = options.WindowLeftTopCoordsX;
	parameters.windowY0 = options.WindowLeftTopCoordsY;
	parameters.windowWidth = options.WindowWidth;
	parameters.windowHeight = options.WindowHeight;

	parameters.truncationLength = options.TruncationLength;
	parameters.truncationMode = !options.Parsing;
	parameters.truncationRate = options.TruncationRate;
	parameters.tier2Threads = options.Tier2Threads;

	parameters.decodePasses = options.DecodePasses;
	parameters.imageInfo = sampleImage;
	if (options.MaxWidth > sampleImage->width || options.MaxHeight > sampleImage->height)
		parameters.maxStreamSize = GetBufferSizeFromSurface(sampleImage->decoderSurfaceFmt, options.MaxWidth, options.MaxHeight);
	else
		parameters.maxStreamSize = sampleImage->streamSize;

	parameters.maxMemoryAvailable = options.MaxMemoryAvailable;
	parameters.sequentialTiles = options.SequentialTiles;
	parameters.enableMemoryReallocation = options.EnableMemoryReallocation;
	
	return FAST_OK;
}
