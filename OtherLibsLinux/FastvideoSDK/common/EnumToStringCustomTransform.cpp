/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "EnumToStringCustomTransform.h"

const char *EnumToString(fastCustomTransformType_t type) {
	switch (type) {
		case FAST_TRANSFORM_NONE:
			return "none";
		case FAST_TRANSFORM_REARRANGE_2X8_TO_4X4:
			return "rearrange 2x8 to 4x4";
		case FAST_TRANSFORM_REARRANGE_2X4_TO_4X4_A:
			return "rearrange 2x4 to 4x4 (version A)";
		case FAST_TRANSFORM_REARRANGE_2X4_TO_4X4_B:
			return "rearrange 2x4 to 4x4 (version B)";
		case FAST_TRANSFORM_REARRANGE_MIX_2X2_AND_2X4_TO_4X2:
			return "rearrange mix 2x2 and 2x4 to 4x2";
		case FAST_TRANSFORM_SPLIT_BLOCK_BY_PIXEL:
			return "split block by pixel";
		case FAST_TRANSFORM_SPLIT_BLOCK_BY_PLANE:
			return "split block by plane";
		case FAST_TRANSFORM_COLORING_BLOCK:
			return "coloring block";
		case FAST_TRANSFORM_COLORING_PLANE:
			return "coloring plane";
		case FAST_TRANSFORM_BAYER_WHITE_BALANCE:
			return "white balance";
		default:
			return "unknown";
	}
}
