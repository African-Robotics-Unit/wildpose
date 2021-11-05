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

#include "HelperSDICommon.hpp"
#include "HelperSDIPacked.hpp"

/////////////////////////////////////////////////
/// SDI helpers (COMMON version)
/////////////////////////////////////////////////

inline unsigned long GetSDIWidthUV(fastSDIFormat_t sdiFmt, unsigned width) {
	switch (sdiFmt) {
		case FAST_SDI_YV12_BT709:
		case FAST_SDI_YV12_BT2020:
		case FAST_SDI_YV12_BT601:
		case FAST_SDI_YV12_BT601_FR:

		case FAST_SDI_420_8_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT2020:

		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601_FR:

		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601_FR:

		case FAST_SDI_422_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT2020:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT601_FR:

		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601_FR:

		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601_FR:
		{
			return uDivUp(width, 2U);
		}

		case FAST_SDI_NV12_BT709:
		case FAST_SDI_NV12_BT2020:
		case FAST_SDI_NV12_BT601:
		case FAST_SDI_NV12_BT601_FR:

		case FAST_SDI_P010_BT709:
		case FAST_SDI_P010_BT2020:
		case FAST_SDI_P010_BT601:
		case FAST_SDI_P010_BT601_FR:
		{
			return uSnapUp(width, 2U);
		}

		case FAST_SDI_422_8_CbYCrY_BT601:
		case FAST_SDI_422_8_CbYCrY_BT601_FR:
		case FAST_SDI_422_8_CbYCrY_BT709:
		case FAST_SDI_422_8_CbYCrY_BT2020:

		case FAST_SDI_422_8_CrYCbY_BT601:
		case FAST_SDI_422_8_CrYCbY_BT601_FR:
		case FAST_SDI_422_8_CrYCbY_BT709:
		case FAST_SDI_422_8_CrYCbY_BT2020:

		{
			return uDivUp(width, 2U);
		}

		case FAST_SDI_444_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT2020:

		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT2020:
		{
			return width;
		}
	}

	return 0;
}

inline unsigned long GetSDIPitchYCommon(fastSDIFormat_t sdiFmt, unsigned width, bool isHost) {
	return AlignSDIPitch(GetSDIValueSize(sdiFmt) * width, isHost);
}

inline unsigned long GetSDIPitchUVCommon(fastSDIFormat_t sdiFmt, unsigned width, bool isHost) {
	return AlignSDIPitch(GetSDIValueSize(sdiFmt) * GetSDIWidthUV(sdiFmt, width), isHost);
}

inline unsigned long GetSDIHeightUV(fastSDIFormat_t sdiFmt, unsigned height) {
	switch (sdiFmt) {
		case FAST_SDI_YV12_BT709:
		case FAST_SDI_YV12_BT2020:
		case FAST_SDI_YV12_BT601:
		case FAST_SDI_YV12_BT601_FR:

		case FAST_SDI_420_8_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT2020:

		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601_FR:

		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601_FR:
		
		case FAST_SDI_NV12_BT709:
		case FAST_SDI_NV12_BT2020:
		case FAST_SDI_NV12_BT601:
		case FAST_SDI_NV12_BT601_FR:

		case FAST_SDI_P010_BT709:
		case FAST_SDI_P010_BT2020:
		case FAST_SDI_P010_BT601:
		case FAST_SDI_P010_BT601_FR:
		{
			return uDivUp(height, 2U);
		}

		case FAST_SDI_422_8_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT2020:

		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601_FR:

		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601_FR:

		case FAST_SDI_444_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT2020:

		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT2020:
		{
			return height;
		}
	}
	return 0;
}

inline unsigned long GetSDIBufferSizeCommonPlanar(fastSDIFormat_t sdiFmt, unsigned width, unsigned height, bool isHost) {
	switch (sdiFmt) {
		case FAST_SDI_YV12_BT709:
		case FAST_SDI_YV12_BT2020:
		case FAST_SDI_YV12_BT601:
		case FAST_SDI_YV12_BT601_FR:

		case FAST_SDI_420_8_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT2020:
		
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601_FR:
		
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601_FR:
		
		case FAST_SDI_422_8_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT2020:

		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601_FR:

		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601_FR:
		
		case FAST_SDI_444_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT2020:
		
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT2020:
		{
			const unsigned long sizeY = GetSDIPitchYCommon(sdiFmt, width, isHost) * height;
			const unsigned long sizeUV = GetSDIPitchUVCommon(sdiFmt, width, isHost) * GetSDIHeightUV(sdiFmt, height);
			return sizeY + sizeUV * 2;
		}

		case FAST_SDI_NV12_BT709:
		case FAST_SDI_NV12_BT2020:
		case FAST_SDI_NV12_BT601:
		case FAST_SDI_NV12_BT601_FR:

		case FAST_SDI_P010_BT709:
		case FAST_SDI_P010_BT2020:
		case FAST_SDI_P010_BT601:
		case FAST_SDI_P010_BT601_FR:
		{
			const unsigned long sizeY = GetSDIPitchYCommon(sdiFmt, width, isHost) * height;
			const unsigned long sizeUV = GetSDIPitchUVCommon(sdiFmt, width, isHost) * GetSDIHeightUV(sdiFmt, height);
			return sizeY + sizeUV;
		}
	}
	return 0;
}

/////////////////////////////////////////////////
/// SDI helpers (HOST version)
/////////////////////////////////////////////////

inline unsigned long GetSDIPitchY(fastSDIFormat_t sdiFmt, unsigned width) {
	return GetSDIPitchYCommon(sdiFmt, width, true);
}

inline unsigned long GetSDIPitchUV(fastSDIFormat_t sdiFmt, unsigned width) {
	return GetSDIPitchUVCommon(sdiFmt, width, true);
}
/////////////////////////////////////////////////
/// SDI helpers (DEVICE version)
/////////////////////////////////////////////////

inline unsigned long GetDeviceSDIPitchY(fastSDIFormat_t sdiFmt, unsigned width) {
	return GetSDIPitchYCommon(sdiFmt, width, false);
}

inline unsigned long GetDeviceSDIPitchUV(fastSDIFormat_t sdiFmt, unsigned width) {
	return GetSDIPitchUVCommon(sdiFmt, width, false);
}

inline bool IsSDIInterleavedUVFormat(fastSDIFormat_t sdiFmt)
{
	switch (sdiFmt) {
		case FAST_SDI_NV12_BT709:
		case FAST_SDI_NV12_BT2020:
		case FAST_SDI_NV12_BT601:
		case FAST_SDI_NV12_BT601_FR:

		case FAST_SDI_P010_BT709:
		case FAST_SDI_P010_BT2020:
		case FAST_SDI_P010_BT601:
		case FAST_SDI_P010_BT601_FR:
			return true;
	}
	return false;
}

inline bool IsSDIFullyPlanarFormat(fastSDIFormat_t sdiFmt)
{
	switch (sdiFmt) {
		case FAST_SDI_YV12_BT709:
		case FAST_SDI_YV12_BT2020:
		case FAST_SDI_YV12_BT601:
		case FAST_SDI_YV12_BT601_FR:

		case FAST_SDI_420_8_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_8_YCbCr_PLANAR_BT2020:

		case FAST_SDI_444_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT2020:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_444_8_YCbCr_PLANAR_BT601_FR:

		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601_FR:

		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT2020:
		
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601_FR:
		
		case FAST_SDI_422_8_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_8_YCbCr_PLANAR_BT2020:

		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601_FR:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT2020:
		
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT709:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT2020:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601:
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601_FR:
			return true;
	}
	return false;
}

/////////////////////////////////////////////////
/// SDI pitch converter (DEVICE <-> HOST)
///		Note: device memory aligned to FAST_ALIGNMENT
///			  host memory not aligned
/////////////////////////////////////////////////
inline void PackDeviceSDIPlanar(
	unsigned char *src,
	unsigned char *dst,

	fastSDIFormat_t sdiFmt,
	unsigned width, unsigned height
) {
	if (IsSDIFullyPlanarFormat(sdiFmt))
	{
		// Y block
		const unsigned srcPitchY = GetDeviceSDIPitchY(sdiFmt, width);
		const unsigned dstPitchY = GetSDIPitchY(sdiFmt, width);
		for (unsigned i = 0; i < height; i++) {
			memcpy(&dst[i * dstPitchY], &src[i * srcPitchY], dstPitchY);
		}

		// U block
		const unsigned srcPitchUV = GetDeviceSDIPitchUV(sdiFmt, width);
		const unsigned dstPitchUV = GetSDIPitchUV(sdiFmt, width);

		unsigned char *srcPtrU = &src[srcPitchY * height];
		unsigned char *dstPtrU = &dst[dstPitchY * height];
		const unsigned heightUV = GetSDIHeightUV(sdiFmt, height);

		for (unsigned i = 0; i < heightUV; i++) {
			memcpy(&dstPtrU[i * dstPitchUV], &srcPtrU[i * srcPitchUV], dstPitchUV);
		}

		// V block
		unsigned char *srcPtrV = &srcPtrU[srcPitchUV * heightUV];
		unsigned char *dstPtrV = &dstPtrU[dstPitchUV * heightUV];

		for (unsigned i = 0; i < heightUV; i++) {
			memcpy(&dstPtrV[i * dstPitchUV], &srcPtrV[i * srcPitchUV], dstPitchUV);
		}
			
	}
	else if (IsSDIInterleavedUVFormat(sdiFmt))
	{
		// Y block
		const unsigned srcPitchY = GetDeviceSDIPitchY(sdiFmt, width);
		const unsigned dstPitchY = GetSDIPitchY(sdiFmt, width);
		for (unsigned i = 0; i < height; i++) {
			memcpy(&dst[i * dstPitchY], &src[i * srcPitchY], dstPitchY);
		}

		// UV block
		const unsigned srcPitchUV = GetDeviceSDIPitchUV(sdiFmt, width);
		const unsigned dstPitchUV = GetSDIPitchUV(sdiFmt, width);

		unsigned char *srcPtrU = &src[srcPitchY * height];
		unsigned char *dstPtrU = &dst[dstPitchY * height];
		const unsigned heightUV = GetSDIHeightUV(sdiFmt, height);

		for (unsigned i = 0; i < heightUV; i++) {
			memcpy(&dstPtrU[i * dstPitchUV], &srcPtrU[i * srcPitchUV], dstPitchUV);
		}
			
	}
	else if (sdiFmt == FAST_SDI_RGBA)
	{
		// just copy data to output
		memcpy(dst, src, sizeof(int)* width * height);
	
	}
	
}

inline void UnPackDeviceSDIPlanar(
	unsigned char *src,
	unsigned char *dst,

	fastSDIFormat_t sdiFmt,
	unsigned width, unsigned height
) {
	if (IsSDIFullyPlanarFormat(sdiFmt))
	{
		// Y block
		const unsigned srcPitchY = GetSDIPitchY(sdiFmt, width);
		const unsigned dstPitchY = GetDeviceSDIPitchY(sdiFmt, width);
		for (unsigned i = 0; i < height; i++) {
			memcpy(&dst[i * dstPitchY], &src[i * srcPitchY], dstPitchY);
		}

		// U block
		const unsigned srcPitchUV = GetSDIPitchUV(sdiFmt, width);
		const unsigned dstPitchUV = GetDeviceSDIPitchUV(sdiFmt, width);
		const unsigned heightUV = GetSDIHeightUV(sdiFmt, height);

		unsigned char *srcPtrU = &src[srcPitchY * height];
		unsigned char *dstPtrU = &dst[dstPitchY * height];

		for (unsigned i = 0; i < heightUV; i++) {
			memcpy(&dstPtrU[i * dstPitchUV], &srcPtrU[i * srcPitchUV], dstPitchUV);
		}

		// V block
		unsigned char *srcPtrV = &srcPtrU[srcPitchUV * heightUV];
		unsigned char *dstPtrV = &dstPtrU[dstPitchUV * heightUV];

		for (unsigned i = 0; i < heightUV; i++) {
			memcpy(&dstPtrV[i * dstPitchUV], &srcPtrV[i * srcPitchUV], dstPitchUV);
		}
	}
	else if (IsSDIInterleavedUVFormat(sdiFmt))
	{
		// Y block
		const unsigned srcPitchY = GetSDIPitchY(sdiFmt, width);
		const unsigned dstPitchY = GetDeviceSDIPitchY(sdiFmt, width);
		for (unsigned i = 0; i < height; i++) {
			memcpy(&dst[i * dstPitchY], &src[i * srcPitchY], dstPitchY);
		}

		// UV block
		const unsigned srcPitchUV = GetSDIPitchUV(sdiFmt, width);
		const unsigned dstPitchUV = GetDeviceSDIPitchUV(sdiFmt, width);
		const unsigned heightUV = GetSDIHeightUV(sdiFmt, height);

		unsigned char *srcPtrU = &src[srcPitchY * height];
		unsigned char *dstPtrU = &dst[dstPitchY * height];

		for (unsigned i = 0; i < heightUV; i++) {
			memcpy(&dstPtrU[i * dstPitchUV], &srcPtrU[i * srcPitchUV], dstPitchUV);
		}
			
	}
}
