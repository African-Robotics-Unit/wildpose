/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __RESIZE_H__
#define __RESIZE_H__

#include <list>
#include <memory>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"

#include "ResizerSampleOptions.h"
#include "Image.h"

class Resizer {
private:
	std::unique_ptr<unsigned char, FastAllocator> byteStream;
	unsigned byteStreamSize;

	fastResizerHandle_t hResizer;

	fastImportFromHostHandle_t hHostToDeviceAdapter;
	fastExportToHostHandle_t hDeviceToHostAdapter;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	fastDeviceSurfaceBufferHandle_t dstBuffer;

	double maxScaleFactor;

	bool folder;
	bool floatMode;
	bool convertToBGR;

	ResizerSampleOptions options;

	bool info;

public:
	Resizer(bool info) { this->info = info; byteStream = NULL; };
	~Resizer(void) { };

	fastStatus_t Init(ResizerSampleOptions &options, const double maxScaleFactor);
	fastStatus_t Resize(std::list< Image<FastAllocator> > &inputImages, const char *outputPattern);
	fastStatus_t Close() const;
};

#endif
