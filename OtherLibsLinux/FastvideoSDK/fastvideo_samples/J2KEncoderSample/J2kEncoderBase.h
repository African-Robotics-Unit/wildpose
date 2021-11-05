/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __J2K_ENCODER_BASE__
#define __J2K_ENCODER_BASE__

#include "fastvideo_sdk.h"
#include "FastAllocator.h"

#include "fastvideo_encoder_j2k.h"

#include "J2kEncoderOptions.h"

class J2kEncoderBase {
protected:
	unsigned maxWidth;
	unsigned maxHeight;
	unsigned batchSize;
	fastSurfaceFormat_t surfaceFmt;

	J2kEncoderOptions options;
	fastEncoderJ2kStaticParameters_t parameters;
	bool info;
	bool folder;

	const double MaximumSizeIncrease = 1.5; // reserve for uncompressible images

public:
	J2kEncoderBase(void);
	~J2kEncoderBase(void);

	fastStatus_t Init(J2kEncoderOptions &options);
};

#endif // __J2K_ENCODER_BASE__
