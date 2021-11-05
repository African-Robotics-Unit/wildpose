/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#pragma once

#include <list>
#include <memory>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "Image.h"
#include "Jpeg2JpegSampleOptions.h"
#include "MultiThreadInfo.hpp"

class Jpeg2Jpeg {
private:
	std::unique_ptr<unsigned char, FastAllocator> h_ResizedJpegStream;
	unsigned resizedJpegStreamSize;

	fastResizerHandle_t hResizer;
	fastJpegDecoderHandle_t hDecoder;
	fastJpegEncoderHandle_t hEncoder;
	fastImageFiltersHandle_t hImageFilterAfter;
	fastImageFiltersHandle_t hImageFilterBefore;
	fastCropHandle_t hCrop;

	fastDeviceSurfaceBufferHandle_t d_decoderBuffer;
	fastDeviceSurfaceBufferHandle_t d_resizerBuffer;
	fastDeviceSurfaceBufferHandle_t d_imageFilterBufferAfter;
	fastDeviceSurfaceBufferHandle_t d_imageFilterBufferBefore;
	fastDeviceSurfaceBufferHandle_t d_cropBuffer;

	Jpeg2JpegSampleOptions options;

	fastJfifInfo_t jfifInfo, outJfifInfo;
	unsigned maxBufferSize;

	double maxScaleFactor;
	unsigned channelCount;

	bool info;
	bool mtMode;

public:
	Jpeg2Jpeg(bool info, bool mtMode = false) {
		this->info = info;
		this->mtMode = mtMode;

		jfifInfo = { };
		outJfifInfo = { };
	};
	~Jpeg2Jpeg(void) { };

	fastStatus_t Init(Jpeg2JpegSampleOptions &options, double maxScaleFactor, MtResult *result = nullptr);
	fastStatus_t Resize(std::list< Bytestream<FastAllocator> > &inputImages, int threadId, MtResult *result = nullptr);
	fastStatus_t Close() const;
};
