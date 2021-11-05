/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __SDI_IMPORT_FROM_HOST__
#define __SDI_IMPORT_FROM_HOST__

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "Image.h"
#include "SDIConverterSampleOptions.h"

class SDIImportFromHost {
private:
	fastSDIImportFromHostHandle_t hImport;
	fastExportToHostHandle_t hExportToHost;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	std::unique_ptr<unsigned char, FastAllocator> h_Result;

	unsigned maxWidth;
	unsigned maxHeight;
	unsigned maxPitch;

	fastSurfaceFormat_t surfaceFmt;
	fastSDIFormat_t sdiFmt;

	bool info;
	bool convertToBGR;
	bool isConvert12to16;

public:
	SDIImportFromHost(bool info) {
		this->info = info;
		hImport = NULL;
		hExportToHost = NULL;
	};
	~SDIImportFromHost(void) {};

	fastStatus_t Init(SDIConverterSampleOptions &options);
	fastStatus_t Transform(Image<FastAllocator > &image, char *outFilename);
	fastStatus_t Transform3(Image<FastAllocator > &img, char *outFilename);
	fastStatus_t Close(void) const;
};

#endif	// __SDI_IMPORT_FROM_HOST__
