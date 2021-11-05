/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FASTVIDEO_J2K_FFMPEG_WRAPPER__
#define __FASTVIDEO_J2K_FFMPEG_WRAPPER__

#include <stdlib.h>

#include "fastvideo_sdk.h"
#include "fastvideo_decoder_j2k.h"
#include "fastvideo_encoder_j2k.h"
#include "libavcodec/avcodec.h"

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
typedef struct fastJ2kDecoderFfmpegWrapperHandleStruct *fastJ2kDecoderFfmpegWrapperHandle_t;
typedef struct fastJ2kEncoderFfmpegWrapperHandleStruct *fastJ2kEncoderFfmpegWrapperHandle_t;

///////////////////////////////////////////////////////////////////////////////
// FFMPEG J2K Encoder functions
///////////////////////////////////////////////////////////////////////////////
typedef struct {
	bool convertTo8Bits;
	bool importFromHost;

	int threadCount;
	int maxBatchSize;

	unsigned width;
	unsigned height;

	// JPEG Encoder params
	bool lossless;
	bool pcrdEnabled;
	bool noMCT;
	
	int overwriteSurfaceBitDepth;
	int outputBitDepth;
	int dwtLevels;
	int codeBlockSize;
	float quality;

    float compressionRatio;
    int tileWidth;
    int tileHeight;
} fastJ2kEncoderFfmpegWrapperOptions_t;

extern fastStatus_t DLL fastJ2kEncoderFfmpegWrapperLibraryInit(fastSdkParametersHandle_t handle);

extern fastStatus_t DLL fastJ2kEncoderFfmpegWrapperCreate(
	fastJ2kEncoderFfmpegWrapperHandle_t* handle,

	enum AVPixelFormat pixelFmt,
	fastJ2kEncoderFfmpegWrapperOptions_t* options
);

extern fastStatus_t DLL fastJ2kEncoderFfmpegWrapperSendFrame(
	fastJ2kEncoderFfmpegWrapperHandle_t handle,

	AVPacket* packet,
	AVFrame* frame
);

extern fastStatus_t DLL fastJ2kEncoderFfmpegWrapperReceivePacket(
	fastJ2kEncoderFfmpegWrapperHandle_t handle,

	AVPacket** packet,
	AVFrame** frame
);

extern fastStatus_t DLL fastJ2kEncoderFfmpegWrapperCheckTheLastPackets(
	fastJ2kEncoderFfmpegWrapperHandle_t handle
);

extern fastStatus_t DLL fastJ2kEncoderFfmpegWrapperDestroy(fastJ2kEncoderFfmpegWrapperHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// FFMPEG J2K Decoder functions
///////////////////////////////////////////////////////////////////////////////

typedef struct {
	bool convertTo8Bits;
	bool convertTo420;
	bool convertToRGB;
	bool exportToHost;

	int threadCount;
	int maxBatchSize;
	int decodePasses;
	unsigned width;
	unsigned height;
} fastJ2kDecoderFfmpegWrapperOptions_t;

extern fastStatus_t DLL fastJ2kDecoderFfmpegWrapperLibraryInit(fastSdkParametersHandle_t handle);

extern fastStatus_t DLL fastJ2kDecoderFfmpegWrapperCreate(
	fastJ2kDecoderFfmpegWrapperHandle_t* handle,
	fastJ2kImageInfo_t* imageInfo,
	fastJ2kDecoderFfmpegWrapperOptions_t* options
);

extern fastStatus_t DLL fastJ2kDecoderFfmpegWrapperDecode(
	fastJ2kDecoderFfmpegWrapperHandle_t handle,

	uint8_t* j2kFrame,
	unsigned j2kFrameSize,

	uint8_t* decodedFrame,
	unsigned decodedFramePitch
);

extern fastStatus_t DLL fastJ2kDecoderFfmpegWrapperDecode3(
	fastJ2kDecoderFfmpegWrapperHandle_t handle,

	uint8_t* j2kFrame,
	const unsigned j2kFrameSize,

	uint8_t* decodedFrameY,
	unsigned decodedFramePitchY,

	uint8_t* decodedFrameCr1,
	unsigned decodedFramePitchCr1,

	uint8_t* decodedFrameCr2,
	unsigned decodedFramePitchCr2
);

extern fastStatus_t DLL fastJ2kDecoderFfmpegWrapperDestroy(fastJ2kDecoderFfmpegWrapperHandle_t handle);

#ifdef __cplusplus
}
#endif

#endif // __FASTVIDEO_J2K_FFMPEG_WRAPPER__