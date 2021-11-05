/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "FfmpegPixelFormatConvertion.hpp"

fastSurfaceFormat_t ConvertToSurfaceFmt(AVPixelFormat pixelFormat) {
	switch (pixelFormat) {
		case AV_PIX_FMT_RGB24:
			return FAST_RGB8;

		case AV_PIX_FMT_RGB48LE:
			return FAST_RGB16;

		case AV_PIX_FMT_YUV420P16LE:
		case AV_PIX_FMT_YUV422P16LE:
		case AV_PIX_FMT_YUV444P16LE:
			return FAST_RGB12;

		case AV_PIX_FMT_YUV422P10LE:
			return FAST_RGB12;

		case AV_PIX_FMT_YUV420P:
		case AV_PIX_FMT_YUV422P:
		case AV_PIX_FMT_YUV444P:
			return FAST_RGB8;

		case AV_PIX_FMT_NV12:
			return FAST_RGB8;

		case AV_PIX_FMT_P010LE:
			return FAST_RGB12;

		default:
			return FAST_I8;
	}
}

fastSDIFormat_t ConvertToSdiFmt(AVPixelFormat pixelFormat) {
	switch (pixelFormat) {
		case AV_PIX_FMT_YUV420P:
			return FAST_SDI_420_8_YCbCr_PLANAR_BT709;
		case AV_PIX_FMT_YUV422P:
			return FAST_SDI_422_8_YCbCr_PLANAR_BT709;
		case AV_PIX_FMT_YUV444P:
			return FAST_SDI_444_8_YCbCr_PLANAR_BT709;

		case AV_PIX_FMT_YUV420P16LE:
			return FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT709;
		case AV_PIX_FMT_YUV422P16LE:
			return FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT709;
		case AV_PIX_FMT_YUV444P16LE:
			return FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT709;

		case AV_PIX_FMT_YUV422P10LE:
			return FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT709;

		case AV_PIX_FMT_NV12:
			return FAST_SDI_NV12_BT709;
		case AV_PIX_FMT_P010LE:
			return FAST_SDI_P010_BT709;

		default:
			return FAST_SDI_RGBA;
	}
}

fastJpegFormat_t ConvertToSubsampling(AVPixelFormat pixelFormat) {
	switch (pixelFormat) {
		case AV_PIX_FMT_RGB24:
		case AV_PIX_FMT_RGB48LE:

		case AV_PIX_FMT_YUV444P16LE:
		case AV_PIX_FMT_YUV444P:
			return FAST_JPEG_444;

		case AV_PIX_FMT_NV12:
		case AV_PIX_FMT_P010LE:

		case AV_PIX_FMT_YUV420P16LE:
		case AV_PIX_FMT_YUV420P:
			return FAST_JPEG_420;

		case AV_PIX_FMT_YUV422P16LE:
		case AV_PIX_FMT_YUV422P:

		case AV_PIX_FMT_YUV422P10LE:
			return FAST_JPEG_422;

		default:
			return FAST_JPEG_444;
	}
}

unsigned ConvertToOutputBitDepth(AVPixelFormat pixelFormat)
{
	switch (pixelFormat) {
		case AV_PIX_FMT_YUV420P:
		case AV_PIX_FMT_YUV422P:
		case AV_PIX_FMT_YUV444P:
		case AV_PIX_FMT_NV12:
			return 0;

		case AV_PIX_FMT_YUV420P16LE:
		case AV_PIX_FMT_YUV422P16LE:
		case AV_PIX_FMT_YUV444P16LE:
		case AV_PIX_FMT_YUV422P10LE:
		case AV_PIX_FMT_P010LE:
			return 10;
		default:
			return 0;
	}

	return 0;
}
