/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __DEBAYER_MUX__
#define __DEBAYER_MUX__

#include <memory>
#include <list>

#include "fastvideo_sdk.h"
#include "FastAllocator.h"

#include "Image.h"
#include "DebayerSampleOptions.h"

class DebayerMux {
private:
	fastImportFromHostHandle_t hImportFromHost;
	fastMuxHandle_t hMux;
	fastImageFiltersHandle_t hSam;
	fastDebayerHandle_t hDebayer;
	fastExportToHostHandle_t hExportToHost;

	/*
	 * [0] - hImportFromHost -> hMad -> hDebayer -> hExportToHost
	 * [1] - hImportFromHost -> hDebayer -> hExportToHost
	 */
	fastDeviceSurfaceBufferHandle_t muxBuffers[2];
	fastDeviceSurfaceBufferHandle_t ddebayerBuffer;
	fastDeviceSurfaceBufferHandle_t dstBuffer;

	std::unique_ptr<unsigned char, FastAllocator> buffer;

	unsigned maxWidth;
	unsigned maxHeight;
	fastSurfaceFormat_t dstSurfaceFmt;

	bool info;
	bool convertToBGR;
	fastBayerPattern_t bayerPattern;

public:
	DebayerMux(bool info) { this->info = info; hSam = NULL; };
	~DebayerMux(void) {};

	fastStatus_t Init(DebayerSampleOptions &options, float *matrixA, char *matrixB);
	fastStatus_t Transform(std::list< Image<FastAllocator> > &image);
	fastStatus_t Close(void);
};

#endif // __DEBAYER_MUX__
