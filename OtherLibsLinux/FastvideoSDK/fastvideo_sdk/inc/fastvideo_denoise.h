/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FAST_DENOISE_H__
#define __FAST_DENOISE_H__

#include "fastvideo_sdk.h"

typedef enum  {
	FAST_THRESHOLD_FUNCTION_UNKNOWN,
	FAST_THRESHOLD_FUNCTION_HARD,
	FAST_THRESHOLD_FUNCTION_SOFT,
	FAST_THRESHOLD_FUNCTION_GARROTE
} fastDenoiseThresholdFunctionType_t;

typedef enum {
	FAST_WAVELET_CDF97,
	FAST_WAVELET_CDF53
} fastWaveletType_t;

#ifdef __cplusplus
    extern "C" {
#endif

#ifdef FAST_EXPORTS
	#define DLL __declspec(dllexport) __cdecl
#else
	#define DLL
#endif

////////////////////////////////////////////////////////////////////////////////
// Basic data types
////////////////////////////////////////////////////////////////////////////////

typedef struct {
	fastDenoiseThresholdFunctionType_t function;
	fastWaveletType_t wavelet;
} denoise_static_parameters_t;

typedef struct {
	int dwt_levels;
	float enhance[3];
	float threshold[3];
	float threshold_per_level[33];
} denoise_parameters_t;

typedef struct fastDenoiseHandleStruct *fastDenoiseHandle_t;

extern fastStatus_t DLL fastDenoiseLibraryInit(fastSdkParametersHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Denoise calls
///////////////////////////////////////////////////////////////////////////////
extern fastStatus_t DLL fastDenoiseCreate(
	fastDenoiseHandle_t *handle,
	fastSurfaceFormat_t surfaceFmt,

	void *staticDenoiseParameters,
	
	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

extern fastStatus_t DLL fastDenoiseGetAllocatedGpuMemorySize(
	fastDenoiseHandle_t handle,
	size_t *requestedGpuSizeInBytes
);

extern fastStatus_t DLL fastDenoiseChangeSrcBuffer(
	fastDenoiseHandle_t	handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

extern fastStatus_t DLL fastDenoiseTransform(
 	fastDenoiseHandle_t handle,
	void *denoiseParameters,

 	unsigned width,
	unsigned height
);

extern fastStatus_t DLL fastDenoiseTransformBayerPlanes(
 	fastDenoiseHandle_t handle,
	void *denoiseParameters,

 	unsigned width, // source image width
	unsigned height // source image height
);

extern fastStatus_t DLL fastDenoiseDestroy(fastDenoiseHandle_t handle);

#ifdef __cplusplus
}
#endif

#endif // __FAST_DENOISE_H__
