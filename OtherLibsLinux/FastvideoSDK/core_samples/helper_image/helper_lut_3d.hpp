/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __HELPER_LUT_3D__
#define __HELPER_LUT_3D__

#include "fastvideo_sdk.h"
#include "SampleTypes.h"

fastStatus_t fvLoadCube3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_r,
	std::unique_ptr<float, FastAllocator> &data_g,
	std::unique_ptr<float, FastAllocator> &data_b,
	unsigned &size
);
fastStatus_t fvSaveCube3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_r,
	std::unique_ptr<float, FastAllocator> &data_g,
	std::unique_ptr<float, FastAllocator> &data_b,
	unsigned size,
	unsigned bitsPerChannel
);

fastStatus_t fvLoadXml3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_h,
	std::unique_ptr<float, FastAllocator> &data_s,
	std::unique_ptr<float, FastAllocator> &data_v,
	fast_uint3* size,
	fastColorSaturationOperationType_t operationType[3]
);
fastStatus_t fvSaveXml3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_h,
	std::unique_ptr<float, FastAllocator> &data_s,
	std::unique_ptr<float, FastAllocator> &data_v,
	fast_uint3 &size,
	fastColorSaturationOperationType_t operationType[3]
);

#endif // __HELPER_LUT__
