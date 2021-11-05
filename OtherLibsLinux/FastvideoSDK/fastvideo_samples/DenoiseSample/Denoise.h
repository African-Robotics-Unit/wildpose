/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __COMPONENTS_DENOISE_H__
#define __COMPONENTS_DENOISE_H__

#include <memory>

#include "fastvideo_sdk.h"
#include "FastAllocator.h"
#include "Image.h"

#include "fastvideo_denoise.h"

#include "DenoiseOptions.h"

class Denoise {
private:
	fastDenoiseHandle_t hDenoise;

	fastImportFromHostHandle_t hHostToDeviceAdapter;
	fastExportToHostHandle_t hDeviceToHostAdapter;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	fastDeviceSurfaceBufferHandle_t dstBuffer;

	std::unique_ptr<unsigned char, FastAllocator> buffer;

	unsigned maxWidth;
	unsigned maxHeight;
	fastSurfaceFormat_t surfaceFmt;

	DenoiseOptions options;
	denoise_static_parameters_t parameters;
	int dwt_levels;
	float threshold[3], enhance[3];
	bool info;
	bool folder;
	bool convertToBGR;

public:
	Denoise(bool info) {
		this->info = info;
		parameters = { };
	}
	~Denoise(void) {}

	fastStatus_t Init(DenoiseOptions &options);
	fastStatus_t Transform(std::list< Image<FastAllocator> > &images);
	fastStatus_t Close(void) const;
};

#endif
