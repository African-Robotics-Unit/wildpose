/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __J2K_ENCODER_ONE_IMAGE__
#define __J2K_ENCODER_ONE_IMAGE__

#include <list>
#include <memory>

#include "fastvideo_sdk.h"
#include "FastAllocator.h"
#include "MallocAllocator.h"
#include "Image.h"

#include "fastvideo_encoder_j2k.h"
#include "J2kEncoderBase.h"

#include "J2kEncoderOptions.h"

class J2kEncoderOneImage : public J2kEncoderBase {
private:
	fastEncoderJ2kHandle_t hEncoder;
	fastImportFromHostHandle_t hHostToDeviceAdapter;

	fastDeviceSurfaceBufferHandle_t srcBuffer;

public:
	J2kEncoderOneImage(bool info);
	~J2kEncoderOneImage(void);

	fastStatus_t Init(J2kEncoderOptions &options);
	fastStatus_t Transform(std::list< Image<FastAllocator> > &images);
	fastStatus_t Close(void) const;
};

#endif // __J2K_ENCODER_ONE_IMAGE__
