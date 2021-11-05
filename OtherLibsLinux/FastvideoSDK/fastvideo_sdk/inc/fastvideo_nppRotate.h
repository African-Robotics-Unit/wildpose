/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FASTVIDEO_NPP_ROTATE__
#define __FASTVIDEO_NPP_ROTATE__

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
typedef struct fastNppRotateHandleStruct *fastNppRotateHandle_t;

typedef struct {
	Point_t leftBottomCorner;
	Point_t leftTopCorner;
	Point_t rightTopCorner;
	Point_t rightBottomCorner;
} NppQuadCorners_t;

extern fastStatus_t DLL fastNppRotateLibraryInit(fastSdkParametersHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// NPP Rotate transform interface
///////////////////////////////////////////////////////////////////////////////
extern fastStatus_t DLL fastNppRotateCreate(
	fastNppRotateHandle_t *handle,
	fastNPPImageInterpolation_t interpolationMode,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

extern fastStatus_t DLL fastNppRotateChangeSrcBuffer(
	fastNppRotateHandle_t	handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

extern fastStatus_t DLL fastNppRotateGetRotateQuad(
	fastNppRotateHandle_t handle,

	unsigned width,
	unsigned height,

	double rotateAngle,

	double shiftX,
	double shiftY,

	NppQuadCorners_t *quadCorners
);

extern fastStatus_t DLL fastNppRotateTransform(
	fastNppRotateHandle_t handle,

	unsigned width,
	unsigned height,

	unsigned dstRoiX,
	unsigned dstRoiY,

	double rotateAngle,

	double shiftX,
	double shiftY
);

extern fastStatus_t DLL fastNppRotateGetAllocatedGpuMemorySize(
	fastNppRotateHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

extern fastStatus_t DLL fastNppRotateDestroy(fastNppRotateHandle_t handle);

#ifdef __cplusplus
}
#endif

#endif // __FASTVIDEO_NPP_ROTATE__