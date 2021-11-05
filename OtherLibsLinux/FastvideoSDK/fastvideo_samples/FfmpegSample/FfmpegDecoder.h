/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FFMPEG_DECODER_H__
#define __FFMPEG_DECODER_H__

#include <memory>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "fastvideo_mjpeg.h"

#include "BaseOptions.h"

class FfmpegDecoder {
private:
	static const unsigned Channels = 3;
	static const fastJpegFormat_t SamplingFmt = FAST_JPEG_420;

	std::unique_ptr<unsigned char, FastAllocator> h_RestoredStream;

	fastJpegDecoderHandle_t hDecoder;
	fastExportToHostHandle_t hDeviceToHostAdapter;
	fastMJpegReaderHandle_t hMjpeg;

	fastDeviceSurfaceBufferHandle_t dstBuffer;

	fastSurfaceFormat_t surfaceFmt;
	fastMJpegStreamInfo_t streamInfo;
	fastJfifInfo_t jfifInfo;

	unsigned pitch;
	char* outputFilePattern;

	bool info;
	bool convertToBGR;

public:
	FfmpegDecoder(bool info) {
		this->info = info;
		hMjpeg = NULL;
		jfifInfo = { };
	};
	~FfmpegDecoder(void) {};

	fastStatus_t Init(BaseOptions &options);
	fastStatus_t Decode(void);
	fastStatus_t Close(void) const;
};

#endif
