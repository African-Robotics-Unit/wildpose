/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __COMPONENTS_HSV_LUT_3D__
#define __COMPONENTS_HSV_LUT_3D__

#include <memory>
#include <list>

#include "fastvideo_sdk.h"
#include "FastAllocator.h"

#include "Image.h"
#include "LutSampleOptions.h"

#include "SampleTypes.h"

class HsvLut3D {
private:
	fastImportFromHostHandle_t hImportFromHost;
	fastImageFiltersHandle_t hLut;
	fastExportToHostHandle_t hExportToHost;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	fastDeviceSurfaceBufferHandle_t lutBuffer;

	std::unique_ptr<float, FastAllocator> lut_H;
	std::unique_ptr<float, FastAllocator> lut_S;
	std::unique_ptr<float, FastAllocator> lut_V;
	std::unique_ptr<unsigned char, FastAllocator> buffer;

	fast_uint3 lutSize;

	unsigned maxWidth;
	unsigned maxHeight;
	fastSurfaceFormat_t dstSurfaceFmt;

	bool info;
	bool folder;
	bool convertToBGR;

public:
	HsvLut3D(bool info) { this->info = info; };
	~HsvLut3D(void) {};

	fastStatus_t Init(
		LutSampleOptions &options,

		float *lut3D_H,
		float *lut3D_S,
		float *lut3D_V,
		fast_uint3 lut3DSize
	);
	fastStatus_t Transform(
		std::list<Image<FastAllocator> > &image,

		float *lut3D_H,
		float *lut3D_S,
		float *lut3D_V,
		fast_uint3 lut3DSize
	);
	fastStatus_t Close(void) const;
};

#endif // __COMPONENTS_HSV_LUT_3D__
