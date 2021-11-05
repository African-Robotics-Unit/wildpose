/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __DECODER_H__
#define __DECODER_H__

#include <list>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"

#include "Image.h"
#include "JpegDecoderSampleOptions.h"
#include "MultiThreadInfo.hpp"

class Decoder {
private:
	fastJpegDecoderHandle_t hDecoder;
	fastExportToHostHandle_t hDeviceToHostAdapter;

	fastDeviceSurfaceBufferHandle_t dstBuffer;

	fastSurfaceFormat_t surfaceFmt;
	fastJfifInfo_t jfifInfo;

	bool info;

	BaseOptions options;
	bool mtMode;

	static inline unsigned uDivUp(unsigned a, unsigned b);

public:
	Decoder(bool info, bool mtMode) {
		this->info = info;
		this->mtMode = mtMode;
		jfifInfo = { };
	};
	~Decoder(void) { };

	fastStatus_t Init(JpegDecoderSampleOptions &options, MtResult *result);
	fastStatus_t Decode(std::list< Bytestream<FastAllocator> > &inputImgs, std::list< Image<FastAllocator> > &outputImgs, int threadId, MtResult *result);
	fastStatus_t Close() const;
};

#endif
