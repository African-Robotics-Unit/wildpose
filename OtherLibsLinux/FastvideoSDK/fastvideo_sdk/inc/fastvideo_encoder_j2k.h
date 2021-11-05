/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#pragma once

#include <stdlib.h>

#include "fastvideo_sdk.h"

typedef enum {
    FAST_ENCODER_J2K_ALGORITHM_UNKNOWN,
    FAST_ENCODER_J2K_ALGORITHM_ENCODE_REVERSIBLE,
    FAST_ENCODER_J2K_ALGORITHM_ENCODE_IRREVERSIBLE
} fastEncoderJ2kAlgorithmType_t;

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

typedef struct
{
    bool lossless;
    bool pcrdEnabled;
    bool noMCT;
    bool yuvSubsampledFormat;
    int overwriteSurfaceBitDepth;
    int outputBitDepth;
    int dwtLevels;
    int codeblockSize;
    float maxQuality;
    float compressionRatio;
    bool info;
    int tileWidth;
    int tileHeight;
    int tier2Threads;
    int ss1_x, ss1_y, ss2_x, ss2_y, ss3_x, ss3_y; // predefined subsampling factors for YUV/YCbCr channels
} fastEncoderJ2kStaticParameters_t;

typedef struct
{
    bool writeHeader;
    float quality;
    long targetStreamSize;
} fastEncoderJ2kDynamicParameters_t;

typedef struct
{
    // duration of every stage
    double s1_preprocessing;
    double s2_dwt;
    double s3_tier1;
    double s4_pcrd;
    double s5_gathering;
    double s6_copy;
    double s7_tier2;
    double s8_write;
    double elapsedTime;
    // other characteristics
    int codeblockCount;
    int maxCodeblockLength;
    size_t copySize;
    size_t outputSize;
} fastEncoderJ2kReport_t;

typedef struct
{
    unsigned char *byteStream;
    size_t bufferSize;
    size_t streamSize;
    bool truncated;
} fastEncoderJ2kOutput_t;

typedef struct fastEncoderJ2kHandleStruct *fastEncoderJ2kHandle_t;

///////////////////////////////////////////////////////////////////////////////
// EncoderJ2k calls
///////////////////////////////////////////////////////////////////////////////

extern fastStatus_t DLL fastEncoderJ2kLibraryInit(fastSdkParametersHandle_t handle);

extern fastStatus_t DLL fastEncoderJ2kCreate(
    fastEncoderJ2kHandle_t *handle,

    fastEncoderJ2kStaticParameters_t *parameters,
    fastSurfaceFormat_t surfaceFmt,
    unsigned maxWidth,
    unsigned maxHeight,
    unsigned maxBatchSize,
    fastDeviceSurfaceBufferHandle_t srcBuffer
);

extern fastStatus_t DLL fastEncoderJ2kIsInitialized(
    fastEncoderJ2kHandle_t handle,

    bool *value
);

extern fastStatus_t DLL fastEncoderJ2kGetAllocatedGpuMemorySize(
    fastEncoderJ2kHandle_t handle,

    size_t *allocatedGpuSizeInBytes
);

extern fastStatus_t DLL fastEncoderJ2kTransform(
    fastEncoderJ2kHandle_t handle,
    fastEncoderJ2kDynamicParameters_t *parameters,
    unsigned width,
    unsigned height,

    fastEncoderJ2kOutput_t *output,
    fastEncoderJ2kReport_t *report
);

extern fastStatus_t DLL fastEncoderJ2kFreeSlotsInBatch(
    fastEncoderJ2kHandle_t handle,
    
    int *value
);

extern fastStatus_t DLL fastEncoderJ2kUnprocessedImagesCount(
    fastEncoderJ2kHandle_t handle,
    
    int *value
);

extern fastStatus_t DLL fastEncoderJ2kAddImageToBatch(
    fastEncoderJ2kHandle_t handle,
    fastEncoderJ2kDynamicParameters_t *parameters,
    unsigned width,
    unsigned height
);

extern fastStatus_t DLL fastEncoderJ2kTransformBatch(
    fastEncoderJ2kHandle_t handle,

    fastEncoderJ2kOutput_t *output,
    fastEncoderJ2kReport_t *report
);

extern fastStatus_t DLL fastEncoderJ2kGetNextEncodedImage(
    fastEncoderJ2kHandle_t handle,

    fastEncoderJ2kOutput_t *output,
    fastEncoderJ2kReport_t *report,
    int *imagesLeft
);

extern fastStatus_t DLL fastEncoderJ2kDestroy(
    fastEncoderJ2kHandle_t handle
);

#ifdef __cplusplus
}
#endif
