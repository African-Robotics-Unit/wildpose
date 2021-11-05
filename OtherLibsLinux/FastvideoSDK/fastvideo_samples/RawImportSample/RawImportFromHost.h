/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __RAW_IMPORT_FROM_HOST__
#define __RAW_IMPORT_FROM_HOST__

#include <memory>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "Image.h"

#include "RawImportSampleOptions.h"

class RawImportFromHost {
private:
	fastRawUnpackerHandle_t hRawUnpacker;
	fastExportToHostHandle_t hDeviceToHostAdapter;

	fastDeviceSurfaceBufferHandle_t srcBuffer;

	bool info;
	fastSurfaceFormat_t surfaceFmt;
	RawImportSampleOptions options;

	std::unique_ptr<unsigned char, FastAllocator> h_Result;

public:
	RawImportFromHost(bool info) {
		this->info = info;
		hRawUnpacker = NULL;
		hDeviceToHostAdapter = NULL;
		h_Result = NULL;
	};
	~RawImportFromHost(void) {};

	fastStatus_t Init(RawImportSampleOptions &options);
	fastStatus_t Transform(Image<FastAllocator> &image);
	fastStatus_t Close(void) const;
};

#endif // __RAW_IMPORT_FROM_HOST__