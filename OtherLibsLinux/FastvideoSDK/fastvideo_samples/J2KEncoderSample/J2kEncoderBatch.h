/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __J2K_ENCODER_BATCH__
#define __J2K_ENCODER_BATCH__

#include <list>
#include <memory>
#include <vector>

#include "fastvideo_sdk.h"
#include "FastAllocator.h"
#include "MallocAllocator.h"
#include "Image.h"

#include "fastvideo_encoder_j2k.h"
#include "J2kEncoderBase.h"

#include "J2kEncoderOptions.h"
#include "MultiThreadInfo.hpp"

class J2kEncoderBatch : public J2kEncoderBase {
private:
	fastEncoderJ2kHandle_t hEncoder;
	fastImportFromHostHandle_t hHostToDeviceAdapter;

	fastDeviceSurfaceBufferHandle_t srcBuffer;
	bool mtMode;
	std::vector<std::string> batchFileName;
public:
	J2kEncoderBatch(bool info, bool mtMode = false);
	~J2kEncoderBatch(void);

	fastStatus_t Init(J2kEncoderOptions &options, MtResult *result = nullptr);
	fastStatus_t Transform(std::list< Image<FastAllocator> > &images, MtResult *result = nullptr);
	fastStatus_t Close(void);
};

#endif // __J2K_ENCODER_BATCH__
