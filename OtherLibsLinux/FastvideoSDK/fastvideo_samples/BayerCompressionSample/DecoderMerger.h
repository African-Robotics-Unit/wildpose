/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __DECODER_MERGER__
#define __DECODER_MERGER__

#include <list>
#include <memory>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "fastvideo_jpegCpuDecoder.h"

#include "BaseOptions.h"
#include "Image.h"

class DecoderMerger {
private:
	fastJpegDecoderHandle_t hDecoder;
	fastJpegCpuDecoderHandle_t hDecoderCpu;
	fastBayerMergerHandle_t hBayerMerger;
	fastExportToHostHandle_t hExportToHost;

	fastDeviceSurfaceBufferHandle_t decoderBuffer;
	fastDeviceSurfaceBufferHandle_t bayerMergerBuffer;

	fastSurfaceFormat_t surfaceFmt;
	BaseOptions options;
	bool info;

	fastJfifInfo_t jfifInfo;
	std::unique_ptr<unsigned char, FastAllocator> h_Result;

public:
	DecoderMerger(bool info) {
		this->info = info;
		hBayerMerger = NULL;
		hDecoder = NULL;
		hDecoderCpu = NULL;
		hExportToHost = NULL;

		jfifInfo = { };
	};
	~DecoderMerger(void) { };

	fastStatus_t Init(BaseOptions &options, unsigned maxRestoredWidth, unsigned maxRestoredHeight);
	fastStatus_t Transform(std::list< Bytestream<FastAllocator> > &inputImages);
	fastStatus_t Close() const;
};

#endif // __DECODER_MERGER__
