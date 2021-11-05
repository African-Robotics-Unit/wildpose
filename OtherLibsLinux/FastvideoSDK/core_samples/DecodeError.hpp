/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __DECODE_ERROR__
#define __DECODE_ERROR__

#include <stdio.h>

#include "fastvideo_sdk.h"

inline bool DecodeError ( fastStatus_t code ) {
	switch( code ) {
		case FAST_TRIAL_PERIOD_EXPIRED:
			fprintf(stderr, "Trial period has been expired\n");
			break;

		case FAST_INVALID_DEVICE:
			fprintf(stderr, "Incompatible device or no CUDA-capable devices present\n");
			break;
		case FAST_INCOMPATIBLE_DEVICE:
			fprintf(stderr, "Incompatible device or no CUDA-capable devices present\n");
			break;
		case FAST_INSUFFICIENT_DEVICE_MEMORY:
			fprintf(stderr, "Image file size exceeds GPU memory limits\n");
			break;
		case FAST_INSUFFICIENT_HOST_MEMORY:
			fprintf(stderr, "Image file size exceeds RAM limits\n");
			break;
		case FAST_INVALID_HANDLE:
			fprintf(stderr, "Invalid handle\n");
			break;
		case FAST_INVALID_VALUE:
			fprintf(stderr, "Invalid API enum value\n");
			break;
		case FAST_UNAPPLICABLE_OPERATION:
			fprintf(stderr, "This operation cannot be applied for current type of data\n");
			break;
		case FAST_INVALID_SIZE:
			fprintf(stderr, "Invalid image dimensions\n");
			break;
		case FAST_UNALIGNED_DATA:
			fprintf(stderr, "Buffer base pointers or pitch not properly aligned\n");
			break;
		case FAST_INVALID_TABLE:
			fprintf(stderr, "Invalid quantization / Huffman table\n");
			break;
		case FAST_BITSTREAM_CORRUPT:
			fprintf(stderr, "Decoding error\n");
			break;
		case FAST_EXECUTION_FAILURE:
			fprintf(stderr, "Device execution failure\n");
			break;
		case FAST_INTERNAL_ERROR:
			fprintf(stderr, "Internal error\n");
			break;

		case FAST_IO_ERROR:
			fprintf(stderr, "Failed to open/access file\n");
			break;
		case FAST_INVALID_FORMAT:
			fprintf(stderr, "Invalid file format\n");
			break;
		case FAST_UNSUPPORTED_FORMAT:
			fprintf(stderr, "File format is unsupported by the current version\n");
			break;
		case FAST_END_OF_STREAM:
			fprintf(stderr, "End of stream\n");
			break;

		case FAST_MJPEG_THREAD_ERROR:
			fprintf(stderr, "Error in child process\n");
			break;
		case FAST_TIMEOUT:
			fprintf(stderr, "Free buffer item waiting error\n");
			break;
		case FAST_MJPEG_OPEN_FILE_ERROR:
			fprintf(stderr, "Open file error\n");
			break;

		case FAST_UNKNOWN_ERROR:
			fprintf(stderr, "Unrecognized error\n");
			break;

		case FAST_OK: break;

		default:
			fprintf(stderr, "Unknown statement\n");
			break;
	}

	return code == FAST_OK;
}

#endif // __DECODE_ERROR__