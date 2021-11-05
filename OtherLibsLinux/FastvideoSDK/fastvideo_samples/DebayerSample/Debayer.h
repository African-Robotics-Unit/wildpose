/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __DEBAYER_H__
#define __DEBAYER_H__

#include <list>
#include <memory>

#include <vector_types.h>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "Image.h"
#include "DebayerSampleOptions.h"
#include "MultiThreadInfo.hpp"

class Debayer {
private:
	fastDebayerHandle_t hDebayer;
	fastImageFiltersHandle_t hSam;
	fastImageFiltersHandle_t hWhiteBalance;

	fastImportFromHostHandle_t hHostToDeviceAdapter;
	fastExportToHostHandle_t hDeviceToHostAdapter;

	fastSurfaceConverterHandle_t hSurfaceConverterTo16;
	fastSurfaceConverterHandle_t hSurfaceConverter16to8;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	fastDeviceSurfaceBufferHandle_t bufferTo16;
	fastDeviceSurfaceBufferHandle_t buffer16to8;
	fastDeviceSurfaceBufferHandle_t debayerBuffer;
	fastDeviceSurfaceBufferHandle_t madBuffer;
	fastDeviceSurfaceBufferHandle_t whiteBalanceBuffer;

	DebayerSampleOptions options;
	uint2 scaleFactor;

	bool convertTo16;
	bool info;
	bool mtMode;
	fastSurfaceFormat_t surfaceFmt;

	std::unique_ptr<unsigned char, FastAllocator> h_Result;

public:
	Debayer(bool info, bool mtMode = false) {
		this->info = info;
		this->mtMode = mtMode;
		hDebayer = NULL;
		hSam = NULL;
		hWhiteBalance = NULL;
		hHostToDeviceAdapter = NULL;
		h_Result = NULL;
	};
	~Debayer(void) { };

	fastStatus_t Init(DebayerSampleOptions &options, float *matrixA, char *matrixB, MtResult *result = nullptr);
	fastStatus_t Transform(std::list< Image<FastAllocator> > &image, int threadId, MtResult *result = nullptr);
	fastStatus_t Close(void) const;
};

#endif