/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "EnumToStringDenoise.h"

const char* EnumToString(fastDenoiseThresholdFunctionType_t value) {
	switch (value) {
	case FAST_THRESHOLD_FUNCTION_UNKNOWN:
		return "Unknown";
	case FAST_THRESHOLD_FUNCTION_HARD:
		return "Hard";
	case FAST_THRESHOLD_FUNCTION_SOFT:
		return "Soft";
	case FAST_THRESHOLD_FUNCTION_GARROTE:
		return "Garrote";
	default:
		return "Other";
	}
}

const char* EnumToString(fastWaveletType_t value) {
	switch (value) {
	case FAST_WAVELET_CDF97:
		return "CDF 9/7";
	case FAST_WAVELET_CDF53:
		return "CDF 5/3";
	default:
		return "Other";
	}
}
