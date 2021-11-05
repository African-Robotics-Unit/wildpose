/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __DEFRINGE_H__
#define __DEFRINGE_H__

#include <list>
#include <memory>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"

#include "Image.h"
#include "DefringeSampleOptions.h"

class Defringe {
private:
	fastImportFromHostHandle_t hImportFromHost;
	fastImageFiltersHandle_t hDefringe;
	fastExportToHostHandle_t hExportToHost;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	fastDeviceSurfaceBufferHandle_t dstBuffer;

	unsigned maxWidth;
	unsigned maxHeight;

	bool info;
	bool folder;
	bool convertToBGR;

	std::unique_ptr<unsigned char, FastAllocator> h_Result;

	fastSurfaceFormat_t surfaceFmt;

public:
	Defringe(bool info) { this->info = info; hDefringe = NULL; };
	~Defringe(void) {};

	fastStatus_t Init(DefringeSampleOptions &options);
	fastStatus_t Transform(std::list< Image<FastAllocator> > &image);
	fastStatus_t Close(void) const;
};

#endif // __DEFRINGE_H__
