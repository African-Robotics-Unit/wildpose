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

#include "HelperSDIPlanar.hpp"
#include "HelperSDIPacked.hpp"

inline unsigned long GetSDIBufferSizeCommon(fastSDIFormat_t sdiFmt, unsigned width, unsigned height, bool isHost)
{
	if (sdiFmt == FAST_SDI_RGBA)
		return sizeof(int) * width * height;
	if (IsPackedSDI(sdiFmt))
		return GetSDIBufferSizeCommonPacked(sdiFmt, width, height, isHost);
	return GetSDIBufferSizeCommonPlanar(sdiFmt, width, height, isHost);
}

/////////////////////////////////////////////////
/// SDI helpers (HOST version)
/////////////////////////////////////////////////

inline unsigned long GetSDIBufferSize(fastSDIFormat_t sdiFmt, unsigned width, unsigned height) {
	return GetSDIBufferSizeCommon(sdiFmt, width, height, true);
}

/////////////////////////////////////////////////
/// SDI helpers (DEVICE version)
/////////////////////////////////////////////////

inline unsigned long GetDeviceSDIBufferSize(fastSDIFormat_t sdiFmt, unsigned width, unsigned height) {
	return GetSDIBufferSizeCommon(sdiFmt, width, height, false);
}

/////////////////////////////////////////////////
/// SDI pitch converter (DEVICE <-> HOST)
///		Note: device memory aligned to FAST_ALIGNMENT
///			  host memory not aligned
/////////////////////////////////////////////////

inline void PackDeviceSDI(
	unsigned char *src,
	unsigned char *dst,

	fastSDIFormat_t sdiFmt,
	unsigned width, unsigned height
) {
	if (sdiFmt == FAST_SDI_RGBA)
	{
		// just copy data to output
		memcpy(dst, src, sizeof(int)* width * height);
	}
	else if (IsPackedSDI(sdiFmt)) {
		PackDeviceSDIPacked(src, dst, sdiFmt, width, height);
	}
	else {
		PackDeviceSDIPlanar(src, dst, sdiFmt, width, height);
	}
}

inline void UnPackDeviceSDI(
	unsigned char *src,
	unsigned char *dst,

	fastSDIFormat_t sdiFmt,
	unsigned width, unsigned height
) {
	if (sdiFmt == FAST_SDI_RGBA) {
		// just copy data to output
		memcpy(dst, src, sizeof(int)* width * height);
	}
	else if (IsPackedSDI(sdiFmt)) {
		UnPackDeviceSDIPacked(src, dst, sdiFmt, width, height);
	}
	else {
		UnPackDeviceSDIPlanar(src, dst, sdiFmt, width, height);
	}
}
