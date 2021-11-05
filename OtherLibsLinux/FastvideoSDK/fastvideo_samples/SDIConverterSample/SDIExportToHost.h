/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __SDI_EXPORT_TO_HOST__
#define __SDI_EXPORT_TO_HOST__

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "Image.h"
#include "SDIConverterSampleOptions.h"

class SDIExportToHost {
private:
	fastImportFromHostHandle_t hImportFromHost;
	fastSDIExportToHostHandle_t hExport;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	std::unique_ptr<unsigned char, FastAllocator> h_Result;

	fastSDIFormat_t sdiFmt;

	unsigned maxWidth;
	unsigned maxHeight;

	bool info;
	
public:
	SDIExportToHost(bool info) {
		this->info = info;
		hImportFromHost = NULL;
		hExport = NULL;
	};
	~SDIExportToHost(void) {};

	fastStatus_t Init(SDIConverterSampleOptions &options);
	fastStatus_t Transform(Image<FastAllocator > &image, char *outFilename) const;
	fastStatus_t Transform3(Image<FastAllocator >& image, char* outFilename) const;
	fastStatus_t Close(void) const;
};

#endif	// __SDI_EXPORT_TO_HOST__
