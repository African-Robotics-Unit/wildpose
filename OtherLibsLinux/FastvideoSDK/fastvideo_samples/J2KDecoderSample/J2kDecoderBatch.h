/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#ifndef __J2K_DECODER_BATCH__
#define __J2K_DECODER_BATCH__

#include <list>

#include "Image.h"

#include "fastvideo_decoder_j2k.h"
#include "FastAllocator.h"

#include "J2kDecoderOptions.h"
#include "J2kDecoderBase.h"
#include "MultiThreadInfo.hpp"

class J2kDecoderBatch : public J2kDecoderBase {
private:
	fastDecoderJ2kHandle_t decoder;
	fastExportToHostHandle_t adapter;

	fastDeviceSurfaceBufferHandle_t buffer;
	fastDeviceSurfaceBufferHandle_t stub;

	bool mtMode;

	fastGpuTimerHandle_t deviceToHostTimer;
	double totalInternalTime;

	fastStatus_t TransformAndExtractBatch(
		Image<FastAllocator>& img,
		std::list< Image<FastAllocator> > *outputImgs
	);

public:
	J2kDecoderBatch(bool mtMode);

	fastStatus_t Init(J2kDecoderOptions &options, fastJ2kImageInfo_t *sampleImage, MtResult *result);
	fastStatus_t Transform(std::list< Bytestream<FastAllocator> > &inputImgs, std::list< Image<FastAllocator> > &outputImgs, MtResult *result);
	fastStatus_t Close() const;
};

#endif // __J2K_DECODER_BATCH__