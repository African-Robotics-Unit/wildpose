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

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "Image.h"
#include "JpegEncoderSampleOptions.h"

class JpegEncoderAsync {
private:
	fastJpegEncoderHandle_t hEncoder;
	fastImportFromHostHandle_t hImportFromHost;
	
	fastDeviceSurfaceBufferHandle_t srcBuffer;

	fastJfifInfo_t jfifInfo;
	fastJfifInfoAsync_t jfifInfoAsync;
	int Quality;

	unsigned maxWidth;
	unsigned maxHeight;

	bool info;
	bool folder;

public:
	JpegEncoderAsync(bool info) {
		this->info = info; hEncoder = NULL; hImportFromHost = NULL; 
		jfifInfo = { };
		jfifInfoAsync = { };
	};
	~JpegEncoderAsync(void) {};

	fastStatus_t Init(JpegEncoderSampleOptions &options);
	fastStatus_t Encode(std::list< Image<FastAllocator> > &inputImgs);
	fastStatus_t Close(void) const;
};
