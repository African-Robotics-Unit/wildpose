/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __PERSPECTIVE_NPP__
#define __PERSPECTIVE_NPP__

#include <list>
#include <memory>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "fastvideo_nppGeometry.h"

#include "Image.h"
#include "NppPerspectiveSampleOptions.h"

class Perspective {
private:
	fastImportFromHostHandle_t hImportFromHost;
	fastNppGeometryHandle_t hPerspective;
	fastExportToHostHandle_t hExportToHost;

	fastSdkParametersHandle_t hSdkParameters;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	fastDeviceSurfaceBufferHandle_t dstBuffer;

	fastSurfaceFormat_t surfaceFmt;
	
	NppPerspectiveSampleOptions options;
	
	bool convertToBGR;

	bool info;

	std::unique_ptr<unsigned char, FastAllocator> h_Result;

public:
	Perspective(bool info) { this->info = info; hPerspective = NULL; };
	~Perspective(void) {};

	fastStatus_t Init(NppPerspectiveSampleOptions &options);
	fastStatus_t Transform(std::list< Image<FastAllocator> > &image);
	fastStatus_t Close() const;
};

#endif // __PERSPECTIVE_NPP__
