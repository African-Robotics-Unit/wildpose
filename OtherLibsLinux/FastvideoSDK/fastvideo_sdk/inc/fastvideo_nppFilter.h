/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FASTVIDEO_NPP_FILTERS__
#define __FASTVIDEO_NPP_FILTERS__

#include "fastvideo_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __GNUC__

#ifdef FAST_EXPORTS
#define DLL __declspec(dllexport) __cdecl
#else
#define DLL
#endif

#else

#define DLL

#endif

////////////////////////////////////////////////////////////////////////////////
// 
////////////////////////////////////////////////////////////////////////////////
typedef struct fastNppFilterHandleStruct *fastNppFilterHandle_t;

typedef enum{
	NPP_GAUSSIAN_SHARPEN = 1,
	NPP_UNSHARP_MASK_SOFT = 2,
	NPP_UNSHARP_MASK_HARD = 3
} fastNPPImageFilterType_t;

typedef struct{
	double radius;
	double sigma;
} fastNPPGaussianFilter_t;

typedef struct {
	float amount;
	float sigma;
	float envelopMedian; /*(0;1)*/
	float envelopSigma; /*(0;), 0.5 - mean median of value interval*/
	int envelopRank; /*2,4,6,8,12*/
	float envelopCoef; /*(;0)*/
	float threshold; /*(0;1)*/
} fastNPPUnsharpMaskFilter_t;

///////////////////////////////////////////////////////////////////////////////
// NPP filters itnerface
///////////////////////////////////////////////////////////////////////////////
extern fastStatus_t DLL fastNppFilterLibraryInit(fastSdkParametersHandle_t handle);

extern fastStatus_t DLL fastNppFilterCreate(
	fastNppFilterHandle_t *handle,

	fastNPPImageFilterType_t filterType,
	void *staticFilterParameters,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

extern fastStatus_t DLL fastNppFilterChangeSrcBuffer(
	fastNppFilterHandle_t	handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

extern fastStatus_t DLL fastNppFilterTransform(
	fastNppFilterHandle_t handle,

	unsigned width,
	unsigned height,

	void *filterParameters
);

extern fastStatus_t DLL fastNppFilterGetAllocatedGpuMemorySize(
	fastNppFilterHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

extern fastStatus_t DLL fastNppFilterDestroy(fastNppFilterHandle_t handle);

#ifdef __cplusplus
}
#endif

#endif // __FASTVIDEO_NPP_FILTERS__