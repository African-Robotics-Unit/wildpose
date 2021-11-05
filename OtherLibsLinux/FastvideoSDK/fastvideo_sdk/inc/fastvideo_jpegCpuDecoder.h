/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FASTVIDEO_JPEG_CPU_DECODER__
#define __FASTVIDEO_JPEG_CPU_DECODER__

#include <stdlib.h>
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
typedef struct fastJpegCpuDecoderHandleStruct *fastJpegCpuDecoderHandle_t;

///////////////////////////////////////////////////////////////////////////////
// JPEG CPU Decoder functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Init JPEG CPU Decoder by SDK global options.

	Init JPEG CPU Decoder by SDK global options.

	\param [in] handle   global options handle
*/
extern fastStatus_t DLL fastJpegCpuDecoderLibraryInit(fastSdkParametersHandle_t handle);

/*! \brief Creates JPEG CPU Decoder and returns associated handle.

	Creates JPEG CPU Decoder and returns associated handle.
	Function fastJpegCpuDecoderCreate allocates all necessary buffers in GPU memory.
	Thus in case GPU does not have enough free memory, then fastJpegDecoderCreate
	returns FAST_INSUFFICIENT_DEVICE_MEMORY.
	Only FAST_RGB12 and FAST_I12 surface formats are supported in other case
	FAST_UNSUPPORTED_SURFACE will be returned.
	Maximum dimensions of the image are set to Decoder during creation. Thus if
	transformation result exceeds the maximum value, then error status FAST_INVALID_SIZE
	will be returned

	\param [out] handle   pointer to created JPEG CPU Decoder
	\param [in] surfaceFmt   type of surface (decoded image). Surface is output for decoder
	\param [in] maxWidth   maximum image width in pixels
	\param [in] maxHeight   maximum image height in pixels
	\param [out] dstBuffer   pointer for linked buffer for next component (output buffer of current component)
*/
extern fastStatus_t DLL fastJpegCpuDecoderCreate(
	fastJpegCpuDecoderHandle_t *handle,

	fastSurfaceFormat_t surfaceFmt,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief Decodes JPEG to surface.

	Decodes JPEG to surface.
	The procedure takes JPEG file from srcJpegStream. There are no additional parameters necessary
	for decoding. Decoded surface is placed to output linked buffer and the following component
	of the pipeline consumes it. Struct fastJfifInfo_t is populated by fastJpegCpuDecode with 
	parameters of decoded jpeg file.
	Fields populated in  fastJfifInfo_t are width, height, bitsPerChannel, huffmanState (for
	luminance and chrominance), quantState (for luminance and chrominance), jpegFmt, restartInterval.
	Decoder returns FAST_IO_ERROR if 8-bit jpeg will be supplied as input.

	\param [in] handle   pointer to JPEG Decoder
	\param [in] srcJpegStream   pointer to buffer with entire jpeg file
	\param [in] jpegStreamSize   buffer size in bytes
	\param [out] jfifInfo   pointer to fastJfifInfo_t struct that takes jpeg parameters of decoded file
*/
extern fastStatus_t DLL fastJpegCpuDecode(
	fastJpegCpuDecoderHandle_t handle,

	unsigned char *srcJpegStream,
	const long jpegStreamSize,

	fastJfifInfo_t *jfifInfo
);

/*! \brief Returns requested GPU memory for JPEG CPU Decoder.

	Returns requested GPU memory for JPEG CPU Decoder.
	Function returns requested memory size in Bytes for JPEG Decoder.

	\param [in] handle   JPEG Cpu Decoder handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastJpegCpuGetAllocatedGpuMemorySize(
	fastJpegCpuDecoderHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Destroys JPEG CPU Decoder.

	Destroys JPEG CPU Decoder.

	\param [in] handle   pointer to JPEG CPU Decoder
*/
fastStatus_t DLL fastJpegCpuDecoderDestroy(fastJpegCpuDecoderHandle_t handle);

#ifdef __cplusplus
}
#endif

#endif // __FASTVIDEO_JPEG_CPU_DECODER__