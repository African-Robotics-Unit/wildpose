/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __MAD16__
#define __MAD16__

#include <list>
#include <memory>

#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "Image.h"
#include "BaseOptions.h"

class Sam {
private:
	fastImportFromHostHandle_t hHostToDevice;
	fastImageFiltersHandle_t hSam;
	fastImageFiltersHandle_t hLut;
	fastExportToHostHandle_t hDeviceToHost;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	fastDeviceSurfaceBufferHandle_t madBuffer;

	unsigned maxWidth;
	unsigned maxHeight;

	bool info;
	bool folder;
	bool convertToBGR;

	std::unique_ptr<unsigned char, FastAllocator> h_Result;

	fastSurfaceFormat_t surfaceFmt;

	bool isTwoByteBlackShiftMatrix;
public:
	Sam(bool info, bool isTwoByteBlackShiftMatrix) {
		this->info = info;
		hSam = NULL;
		hLut = NULL;
		hHostToDevice = NULL;
		hDeviceToHost = NULL;
		h_Result = NULL;
		this->isTwoByteBlackShiftMatrix = isTwoByteBlackShiftMatrix;
	};
	~Sam(void) {};
	void PopulateLinerLut(unsigned short *lut, unsigned lutSize, unsigned newMaxValue);
	fastStatus_t Init(BaseOptions &options, float *matrixA, void *matrixB);
	fastStatus_t Transform(std::list<Image<FastAllocator>> &image);
	fastStatus_t Close(void) const;
};

#endif // __MAD16__
