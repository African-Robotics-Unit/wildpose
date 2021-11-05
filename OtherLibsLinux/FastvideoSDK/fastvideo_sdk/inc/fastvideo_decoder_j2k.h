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

typedef enum 
{
	PT_LRCP = 0, // Layer/Resolution/Component/Position progressive bit stream
	PT_RLCP = 1, // Resolution/Layer/Component/Position progressive bit stream
	PT_RPCL = 2, // Resolution/Position/Component/Layer progressive bit stream
	PT_PCRL = 3, // Position/Component/Resolution/Layer progressive bit stream
	PT_CPRL = 4  // Component/Position/Resolution/Layer progressive bit stream
} ProgressionType;

typedef enum
{
	WT_CDF97 = 0,
	WT_CDF53 = 1,
	WT_CUSTOM = -1
} WaveletType;

// Type of Multi-Component Transformation
typedef enum
{
	MCT_None = 0,
	MCT_Reversible = 1,
	MCT_Irreversible = 2
} MCT_Type;


typedef struct
{
	int bitDepth;
	int subsamplingX;
	int subsamplingY;
} fastJ2kComponentInfo_t;

typedef enum
{
	JPEG2000_CAPABILITY_ANY = 0,
	JPEG2000_CAPABILITY_CSTREAM_RESTRICTION_0,
	JPEG2000_CAPABILITY_CSTREAM_RESTRICTION_1,
	JPEG2000_CAPABILITY_DCINEMA_2K,
	JPEG2000_CAPABILITY_DCINEMA_4K,
	JPEG2000_CAPABILITY_SCALABLE_DCINEMA_2K,
	JPEG2000_CAPABILITY_SCALABLE_DCINEMA_4K,
	JPEG2000_CAPABILITY_OTHER
} J2kCapability_t;

typedef struct {
	unsigned long  Data1;
	unsigned short Data2;
	unsigned short Data3;
	unsigned char  Data4[8];
} fastJp2Uuid_t;

typedef struct
{
	fastJp2Uuid_t id;
	unsigned int dataLength;
	unsigned char *data;
} fastJp2UuidBox_t;

typedef struct 
{
	unsigned int idCount;
	fastJp2Uuid_t *IDs;
	unsigned char urlVersion;
	unsigned int urlFlags;
	unsigned int urlLength;
	char *url;
} fastJp2UuidInfoBox_t;

typedef struct
{
	unsigned int maskLength; // number of bytes used for the compatibility masks (FUAM, DCM, standard and vendor masks)
	unsigned long long fuamMask; // Fully Understand Aspects mask
	unsigned long long dcmMask; // Decode Completely mask
	unsigned int standardFlagsCount;
	unsigned short *standardFlags;
	unsigned long long *standardMasks;
	unsigned int vendorFeatureCount;
	fastJp2Uuid_t *vendorFeatures;
	unsigned long long *vendorMasks;
} fastJp2ReaderRequirementBox_t;

typedef struct fastJp2AssociationBox_t
{
	bool isChild;
	unsigned int childrenCount;
	struct fastJp2AssociationBox_t** children;
	unsigned int labelCount;
	unsigned int *labelLengths;
	char **labels;
	unsigned int xmlCount;
	unsigned int *xmlLengths;
	char **XMLs;
}  fastJp2AssociationBox_t;

typedef struct
{
	fastSurfaceFormat_t decoderSurfaceFmt;
	J2kCapability_t capabilities;
	unsigned width;
	unsigned height;
	unsigned tileWidth;
	unsigned tileHeight;
	unsigned codeblockWidth;
	unsigned codeblockHeight;
	unsigned resolutionLevels;
	size_t streamSize;
	bool subsamplingUsed;
	int componentCount;
	fastJ2kComponentInfo_t *components;

	bool containsRreqBox;
	fastJp2ReaderRequirementBox_t rreq;
	unsigned asocBoxesCount;
	fastJp2AssociationBox_t *asoc; // all association boxes are stored here including children, but they have "is_child" flag enabled.
	unsigned uuidBoxesCount;
	fastJp2UuidBox_t *uuidBoxes;
	bool containsUuidInfoBox;
	fastJp2UuidInfoBox_t uuidInfo;
} fastJ2kImageInfo_t;

typedef struct
{
    int verboseLevel;
	unsigned maxTileWidth;
	unsigned maxTileHeight;
	int resolutionLevels;
	int decodePasses;
	size_t maxStreamSize;
	size_t maxMemoryAvailable;

    // Tier-2
	int tier2Threads;
    bool truncationMode;
    float truncationRate;
	size_t truncationLength;

	// Window
	int windowX0;
	int windowY0;
	int windowWidth;
	int windowHeight;

	bool enableROI;
	bool enableMemoryReallocation;
	bool sequentialTiles;

	fastJ2kImageInfo_t *imageInfo;
} fastDecoderJ2kStaticParameters_t;

typedef struct
{
    // duration of every stage
    double s0_init;
    double s1_tier2;
    double s2_copy;
    double s3_tier1;
    double s4_roi;
    double s5_dequantize;
    double s6_dwt;
    double s7_postprocessing;

    double elapsedTime;

    // other characteristics
    int codeblockCount;
    int copyToGpu_size, copyToHost_size;
	size_t inStreamSize, outStreamSize;

    // output image parameters
    int width, height, channels, bitsPerChannel;
    int tileCount, tilesX, tilesY;
    int resolutionLevels, cbX, cbY;
	ProgressionType progressionType;
	WaveletType dwtType;
	MCT_Type mctType;
} fastDecoderJ2kReport_t;



typedef struct fastDecoderJ2kHandleStruct *fastDecoderJ2kHandle_t;

///////////////////////////////////////////////////////////////////////////////
// DecoderJ2k calls
///////////////////////////////////////////////////////////////////////////////

extern fastStatus_t DLL fastDecoderJ2kPredecode(
	fastJ2kImageInfo_t *imageInfo,
	unsigned char *byteStream,
	size_t streamSize
);


extern fastStatus_t DLL fastDecoderJ2kLibraryInit(fastSdkParametersHandle_t handle);

extern fastStatus_t DLL fastDecoderJ2kCreate(
    fastDecoderJ2kHandle_t *handle,

    fastDecoderJ2kStaticParameters_t *staticParameters,
    fastSurfaceFormat_t surfaceFmt,
    unsigned maxWidth,
    unsigned maxHeight,
    unsigned maxBatchSize,
    fastDeviceSurfaceBufferHandle_t *dstBuffer
);

extern fastStatus_t DLL fastDecoderJ2kIsInitialized(
    fastDecoderJ2kHandle_t handle,

    bool *value
);

extern fastStatus_t DLL fastDecoderJ2kGetAllocatedGpuMemorySize(
    fastDecoderJ2kHandle_t handle,

	size_t *allocatedGpuSizeInBytes
);

extern fastStatus_t DLL fastDecoderJ2kTransform(
    fastDecoderJ2kHandle_t handle,
	unsigned char *byteStream,
	size_t streamSize,

    fastDecoderJ2kReport_t *report
);

extern fastStatus_t DLL fastDecoderJ2kFreeSlotsInBatch(
    fastDecoderJ2kHandle_t handle,
    
    int *value
);

extern fastStatus_t DLL fastDecoderJ2kUnprocessedImagesCount(
    fastDecoderJ2kHandle_t handle,
    
    int *value
);

extern fastStatus_t DLL fastDecoderJ2kAddImageToBatch(
    fastDecoderJ2kHandle_t handle,
	unsigned char *byteStream,
	size_t streamSize
);

extern fastStatus_t DLL fastDecoderJ2kTransformBatch(
    fastDecoderJ2kHandle_t handle,

    fastDecoderJ2kReport_t *report
);

extern fastStatus_t DLL fastDecoderJ2kGetNextDecodedImage(
    fastDecoderJ2kHandle_t handle, 

    fastDecoderJ2kReport_t *report,
    int *imagesLeft
);

extern fastStatus_t DLL fastDecoderJ2kDestroy(
    fastDecoderJ2kHandle_t handle
);

#ifdef __cplusplus
}
#endif
