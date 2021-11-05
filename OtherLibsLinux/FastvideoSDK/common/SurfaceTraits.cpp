/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "SurfaceTraits.hpp"

size_t GetPitchFromSurface(const fastSurfaceFormat_t surfaceFmt, const unsigned width) {
	switch (surfaceFmt) {
	case FAST_I8:
		return SurfaceTrait<FAST_I8>::Pitch(width);
	case FAST_I12:
		return SurfaceTrait<FAST_I12>::Pitch(width);
	case FAST_I16:
		return SurfaceTrait<FAST_I16>::Pitch(width);
	case FAST_I10:
		return SurfaceTrait<FAST_I10>::Pitch(width);
	case FAST_I14:
		return SurfaceTrait<FAST_I14>::Pitch(width);
	case FAST_BGR8:
		return SurfaceTrait<FAST_BGR8>::Pitch(width);
	case FAST_RGB8:
		return SurfaceTrait<FAST_RGB8>::Pitch(width);
	case FAST_BGRX8:
		return SurfaceTrait<FAST_BGRX8>::Pitch(width);
	case FAST_RGB12:
		return SurfaceTrait<FAST_RGB12>::Pitch(width);
	case FAST_RGB16:
		return SurfaceTrait<FAST_RGB16>::Pitch(width);
	default:
		return GetPitchFromSurfaceInternal(surfaceFmt, width);
	}
}

unsigned GetBytesPerChannelFromSurface(const fastSurfaceFormat_t surfaceFmt) {
	switch (surfaceFmt) {
	case FAST_I8:
		return SurfaceTrait<FAST_I8>::GetBytesPerChannel();
	case FAST_I12:
		return SurfaceTrait<FAST_I12>::GetBytesPerChannel();
	case FAST_I16:
		return SurfaceTrait<FAST_I16>::GetBytesPerChannel();
	case FAST_I10:
		return SurfaceTrait<FAST_I10>::GetBytesPerChannel();
	case FAST_I14:
		return SurfaceTrait<FAST_I14>::GetBytesPerChannel();
	case FAST_BGR8:
		return SurfaceTrait<FAST_BGR8>::GetBytesPerChannel();
	case FAST_RGB8:
		return SurfaceTrait<FAST_RGB8>::GetBytesPerChannel();
	case FAST_BGRX8:
		return SurfaceTrait<FAST_BGRX8>::GetBytesPerChannel();
	case FAST_RGB12:
		return SurfaceTrait<FAST_RGB12>::GetBytesPerChannel();
	case FAST_RGB16:
		return SurfaceTrait<FAST_RGB16>::GetBytesPerChannel();
	default:
		return GetBytesPerChannelFromSurfaceInternal(surfaceFmt);
	}
}

unsigned GetNumberOfChannelsFromSurface(const fastSurfaceFormat_t surfaceFmt) {
	switch (surfaceFmt) {
	case FAST_I8:
		return SurfaceTrait<FAST_I8>::NumberOfChannels;
	case FAST_I12:
		return SurfaceTrait<FAST_I12>::NumberOfChannels;
	case FAST_I16:
		return SurfaceTrait<FAST_I16>::NumberOfChannels;
	case FAST_I10:
		return SurfaceTrait<FAST_I10>::NumberOfChannels;
	case FAST_I14:
		return SurfaceTrait<FAST_I14>::NumberOfChannels;
	case FAST_BGR8:
		return SurfaceTrait<FAST_BGR8>::NumberOfChannels;
	case FAST_RGB8:
		return SurfaceTrait<FAST_RGB8>::NumberOfChannels;
	case FAST_BGRX8:
		return SurfaceTrait<FAST_BGRX8>::NumberOfChannels;
	case FAST_RGB12:
		return SurfaceTrait<FAST_RGB12>::NumberOfChannels;
	case FAST_RGB16:
		return SurfaceTrait<FAST_RGB16>::NumberOfChannels;
	default:
		return GetNumberOfChannelsFromSurfaceInternal(surfaceFmt);
	}
}

unsigned GetBitsPerChannelFromSurface(const fastSurfaceFormat_t surfaceFmt) {
	switch (surfaceFmt) {
	case FAST_I8:
		return SurfaceTrait<FAST_I8>::BitsPerChannel;
	case FAST_I12:
		return SurfaceTrait<FAST_I12>::BitsPerChannel;
	case FAST_I16:
		return SurfaceTrait<FAST_I16>::BitsPerChannel;
	case FAST_I10:
		return SurfaceTrait<FAST_I10>::BitsPerChannel;
	case FAST_I14:
		return SurfaceTrait<FAST_I14>::BitsPerChannel;
	case FAST_BGR8:
		return SurfaceTrait<FAST_BGR8>::BitsPerChannel;
	case FAST_RGB8:
		return SurfaceTrait<FAST_RGB8>::BitsPerChannel;
	case FAST_BGRX8:
		return SurfaceTrait<FAST_BGRX8>::BitsPerChannel;
	case FAST_RGB12:
		return SurfaceTrait<FAST_RGB12>::BitsPerChannel;
	case FAST_RGB16:
		return SurfaceTrait<FAST_RGB16>::BitsPerChannel;
	default:
		return GetBitsPerChannelFromSurfaceInternal(surfaceFmt);
	}
}

size_t GetBufferSizeFromSurface(const fastSurfaceFormat_t surfaceFmt, const unsigned width, const unsigned height) {
	switch (surfaceFmt) {
		case FAST_I8:
		case FAST_I10:
		case FAST_I12:
		case FAST_I14:
		case FAST_I16:

		case FAST_BGR8:
		case FAST_RGB8:
		case FAST_BGRX8:

		case FAST_RGB12:
		case FAST_RGB16:
			return GetPitchFromSurface(surfaceFmt, width) * (size_t)height;
		default:
			return GetBufferSizeFromSurfaceInternal(surfaceFmt, width, height);
		}
}
