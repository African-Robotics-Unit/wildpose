/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __DECODER_CPU__
#define __DECODER_CPU__

#include <list>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "fastvideo_jpegCpuDecoder.h"

#include "Image.h"
#include "JpegDecoderSampleOptions.h"

class DecoderCpu {
private:
	fastJpegCpuDecoderHandle_t hDecoder;
	fastExportToHostHandle_t hDeviceToHostAdapter;

	fastDeviceSurfaceBufferHandle_t dstBuffer;

	fastJfifInfo_t jfifInfo;
	BaseOptions options;

	fastSurfaceFormat_t surfaceFmt;
	bool info;

	static inline unsigned uDivUp(unsigned a, unsigned b);

public:
	DecoderCpu(bool info) {
		this->info = info;
		jfifInfo = { };
	};
	~DecoderCpu(void) { };

	fastStatus_t Init(JpegDecoderSampleOptions &options);
	fastStatus_t Decode(std::list< Bytestream< FastAllocator > > &inputImgs, std::list< Image<FastAllocator> > &outputImgs);
	fastStatus_t Close() const;
};

#endif // __DECODER_CPU__
