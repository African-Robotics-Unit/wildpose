/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FASTVIDEO_NPP_GEOMETRY__
#define __FASTVIDEO_NPP_GEOMETRY__

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
typedef struct fastNppGeometryHandleStruct *fastNppGeometryHandle_t;

typedef enum {
	FAST_NPP_GEOMETRY_REMAP,
	FAST_NPP_GEOMETRY_REMAP3,
	FAST_NPP_GEOMETRY_PERSPECTIVE
} fastNppGeometryTransformationType_t;

extern fastStatus_t DLL fastNppGeometryLibraryInit(fastSdkParametersHandle_t handle);

typedef struct {
	float *mapX;
	float *mapY;

	unsigned dstWidth;
	unsigned dstHeight;
} fastNPPRemapMap_t;

typedef struct {
	unsigned R;
	unsigned G;
	unsigned B;
	bool isEnabled;
} fastNPPRemapBackground_t;

typedef struct {
	fastNPPRemapMap_t *map;
	fastNPPRemapBackground_t *background;
} fastNPPRemap_t;

typedef struct {
	fastNPPRemapMap_t *map[3];
	fastNPPRemapBackground_t *background;
} fastNPPRemap3_t;

typedef struct {
	double coeffs[3][3];
} fastNPPPerspective_t;

///////////////////////////////////////////////////////////////////////////////
// NPP Geometry transform interface
///////////////////////////////////////////////////////////////////////////////
extern fastStatus_t DLL fastNppGeometryCreate(
	fastNppGeometryHandle_t *handle,
	fastNppGeometryTransformationType_t transformationType,
	fastNPPImageInterpolation_t interpolationMode,

	void *filterParameters,

	unsigned maxDstWidth,
	unsigned maxDstHeight,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

extern fastStatus_t DLL fastNppGeometryChangeSrcBuffer(
	fastNppGeometryHandle_t	handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

extern fastStatus_t DLL fastNppGeometryTransform(
	fastNppGeometryHandle_t handle,

	void *filterParameters,

	unsigned width,
	unsigned height
);

extern fastStatus_t DLL fastNppGeometryGetAllocatedGpuMemorySize(
	fastNppGeometryHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

extern fastStatus_t DLL fastNppGeometryDestroy(fastNppGeometryHandle_t handle);

#ifdef __cplusplus
}
#endif

#endif // __FASTVIDEO_NPP_GEOMETRY__