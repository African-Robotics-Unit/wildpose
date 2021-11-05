/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FASTVIDEO_NPP_RESIZE__
#define __FASTVIDEO_NPP_RESIZE__

#include <stdlib.h>
#include "fastvideo_nppCommon.h"

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
typedef struct fastNppResizeHandleStruct *fastNppResizeHandle_t;

///////////////////////////////////////////////////////////////////////////////
// NPP Resizes itnerface
///////////////////////////////////////////////////////////////////////////////
extern fastStatus_t DLL fastNppResizeLibraryInit(fastSdkParametersHandle_t handle);

extern fastStatus_t DLL fastNppResizeCreate(
	fastNppResizeHandle_t *handle,

	unsigned resizedWidth,
	unsigned resizedHeight,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

extern fastStatus_t DLL fastNppResizeChangeSrcBuffer(
	fastNppResizeHandle_t	handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

extern fastStatus_t DLL fastNppResizeTransform(
	fastNppResizeHandle_t handle,

	fastNPPImageInterpolation_t resizeType,

	unsigned width,
	unsigned height,

	unsigned resizedWidth,
	unsigned *resizedHeight,

	double shiftX,
	double shiftY
);

extern fastStatus_t DLL fastNppResizeTransformStretch(
	fastNppResizeHandle_t handle,

	fastNPPImageInterpolation_t resizeType,

	unsigned width,
	unsigned height,

	unsigned resizedWidth,
	unsigned resizedHeight,

	double shiftX,
	double shiftY
);

extern fastStatus_t DLL fastNppResizeGetAllocatedGpuMemorySize(
	fastNppResizeHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

extern fastStatus_t DLL fastNppResizeDestroy(fastNppResizeHandle_t handle);

#ifdef __cplusplus
}
#endif

#endif // __FASTVIDEO_NPP_RESIZE__