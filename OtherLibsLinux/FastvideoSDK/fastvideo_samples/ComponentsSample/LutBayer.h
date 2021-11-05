/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __COMPONENTS_LUT_BAYER_H__
#define __COMPONENTS_LUT_BAYER_H__

#include <memory>
#include <list>

#include "fastvideo_sdk.h"
#include "FastAllocator.h"

#include "Image.h"
#include "LutDebayerSampleOptions.h"

class LutBayer {
private:
	fastImportFromHostHandle_t hHostToDeviceAdapter;
	fastImageFiltersHandle_t hLut;
	fastExportToHostHandle_t hDeviceToHostAdapter;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	fastDeviceSurfaceBufferHandle_t lutBuffer;

	std::unique_ptr<unsigned char, FastAllocator> buffer;

	unsigned maxWidth;
	unsigned maxHeight;

	fastSurfaceFormat_t surfaceFmt;
	fastImageFilterType_t filterType;

	fastBayerPattern_t pattern;

	void *lutParameters;

	bool info;
	bool folder;
	bool convertToBGR;

	void *InitParameters(void *R, void *G, void *B);

public:
	LutBayer(bool info) { this->info = info; lutParameters = NULL; };
	~LutBayer(void) {};

	fastStatus_t Init(
		LutDebayerSampleOptions &options,

		void *lut_R,
		void *lut_G,
		void *lut_B
	);
	fastStatus_t Transform(
		std::list< Image<FastAllocator> > &image,

		void *lut_R,
		void *lut_G,
		void *lut_B
	);
	fastStatus_t Close(void) const;
};

#endif // __COMPONENTS_LUT_BAYER_H__
