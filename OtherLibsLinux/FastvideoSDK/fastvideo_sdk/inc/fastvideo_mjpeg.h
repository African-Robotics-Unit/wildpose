/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FASTVIDEO_MJPEG__
#define __FASTVIDEO_MJPEG__

#include <stdlib.h>

#include "fastvideo_sdk.h"

#ifdef __cplusplus
    extern "C" {
#include "libavutil/pixfmt.h"
#include "libavcodec/avcodec.h"
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
typedef struct fastMJpegReaderHandleStruct *fastMJpegReaderHandle_t;
typedef struct fastMJpegWriterHandleStruct *fastMJpegWriterHandle_t;
typedef struct fastMJpegAsyncWriterHandleStruct *fastMJpegAsyncWriterHandle_t;

typedef struct {
	char *fileName;

	fastJpegFormat_t samplingFmt;

	unsigned width;
	unsigned height;
} fastMJpegFileDescriptor_t;

typedef struct {
	char *fileName;
	fastStatus_t resultStatus;
} fastMJpegError_t;

typedef struct {
	int width;
	int height;
	int totalFrames;
	AVPixelFormat pixelFormat;
	AVCodecID     codec_id;
} fastMJpegStreamInfo_t;

extern fastStatus_t DLL fastMJpegLibraryInit(fastSdkParametersHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Motion JPEG Reader
///////////////////////////////////////////////////////////////////////////////
extern fastStatus_t DLL fastMJpegReaderCreate(
	fastMJpegReaderHandle_t *handle,

	char *fileName,
	fastMJpegStreamInfo_t *streamInfo
);

extern fastStatus_t DLL fastMJpegReaderGetNextFrame(
	fastMJpegReaderHandle_t handle,
 
	unsigned char **dst,
	unsigned *dstSize
);

extern fastStatus_t DLL fastMJpegReaderGetFrame(
	fastMJpegReaderHandle_t handle,

	unsigned char **dst,
	unsigned *dstSize
);

extern fastStatus_t DLL fastMJpegReaderClose(fastMJpegReaderHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Motion JPEG Writer
///////////////////////////////////////////////////////////////////////////////
extern fastStatus_t DLL fastMJpegWriterCreate(
	fastMJpegWriterHandle_t *handle,

	fastMJpegFileDescriptor_t *fileDescriptor,
	int frameRate
);

extern fastStatus_t DLL fastMJpegWriteFrame(
	fastMJpegWriterHandle_t handle,
 
	unsigned char *buffer,
	int bufferSize
);

extern fastStatus_t DLL fastMJpegWriterClose(fastMJpegWriterHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Motion JPEG Asynchronous Writer
///////////////////////////////////////////////////////////////////////////////
extern fastStatus_t DLL fastMJpegAsyncWriterCreate(
	fastMJpegAsyncWriterHandle_t *handle,

	int maxWidth,
	int maxHeight,
	int frameRate,

	int workItemQueueLength,
	int maxFilesCount
);

extern fastStatus_t DLL fastMJpegAsyncWriteFrame(
	fastMJpegAsyncWriterHandle_t handle,
 
	fastJfifInfo_t *jfifInfo,
	int fileId
);

extern fastStatus_t DLL fastMJpegAsyncWriterOpenFile(
	fastMJpegAsyncWriterHandle_t handle,

	fastMJpegFileDescriptor_t *fileDescriptor,
	int *fileId
);

extern fastStatus_t DLL fastMJpegAsyncWriterCloseFile(
	fastMJpegAsyncWriterHandle_t handle,
	int fileId
);

extern fastStatus_t DLL fastMJpegAsyncWriterGetJfifInfo(
	fastMJpegAsyncWriterHandle_t handle,

	fastJfifInfo_t **jfifInfo
);

extern fastStatus_t DLL fastMJpegAsyncWriterGetErrorStatus(
	fastMJpegAsyncWriterHandle_t handle,
	fastMJpegError_t **errorStatus
);

extern fastStatus_t DLL fastMJpegAsyncWriterClose(fastMJpegAsyncWriterHandle_t handle);

#ifdef __cplusplus
}
#endif

#endif // __FASTVIDEO_MJPEG__
