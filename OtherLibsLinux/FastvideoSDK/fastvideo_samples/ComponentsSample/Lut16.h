/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __COMPONENTS_LUT_16__
#define __COMPONENTS_LUT_16__

#include <memory>
#include <list>

#include "fastvideo_sdk.h"
#include "FastAllocator.h"

#include "Image.h"
#include "LutSampleOptions.h"

class Lut16 {
private:
	fastImageFiltersHandle_t hLut;
	
	fastImportFromHostHandle_t hImportFromHost;
	fastExportToHostHandle_t hExportToHost;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	fastDeviceSurfaceBufferHandle_t lutBuffer;

	std::unique_ptr<unsigned char, FastAllocator> buffer;

	unsigned maxWidth;
	unsigned maxHeight;
	fastSurfaceFormat_t dstSurfaceFmt;

	void *lutParameters;
	fastImageFilterType_t filterType;
	unsigned RepeatCount;

	bool info;
	bool folder;
	bool convertToBGR;

public:
	Lut16(bool info) { this->info = info; lutParameters = NULL; };
	~Lut16(void) {};

	fastStatus_t Init(
		LutSampleOptions &options,

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
	fastStatus_t Close() const;
};

#endif // __COMPONENTS_LUT_16__
