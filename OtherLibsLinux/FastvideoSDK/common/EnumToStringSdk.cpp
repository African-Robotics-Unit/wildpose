/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "EnumToStringSdk.h"

const char* EnumToString(fastSDIFormat_t value) {
	switch (value) {
		case FAST_SDI_YV12_BT601_FR:
			return "YV12 BT.601 (full range)";
		case FAST_SDI_YV12_BT601:
			return "YV12 BT.601";
		case FAST_SDI_YV12_BT709:
			return "YV12 BT.709";
		case FAST_SDI_YV12_BT2020:
			return "YV12 BT.2020";
		
		case FAST_SDI_420_8_YCbCr_PLANAR_BT601_FR:
			return "YCbCr420 BT.601 (full range)";
		case FAST_SDI_420_8_YCbCr_PLANAR_BT601:
			return "YCbCr420 BT.601";
		case FAST_SDI_420_8_YCbCr_PLANAR_BT709:
			return "YCbCr420 BT.709";
		case FAST_SDI_420_8_YCbCr_PLANAR_BT2020:
			return "YCbCr420 BT.2020";
		
		case FAST_SDI_NV12_BT601_FR:
			return "NV12 BT.601 (full range)";
		case FAST_SDI_NV12_BT601:
			return "NV12 BT.601";
		case FAST_SDI_NV12_BT709:
			return "NV12 BT.709";
		case FAST_SDI_NV12_BT2020:
			return "NV12 BT.2020";
		
		case FAST_SDI_P010_BT601_FR:
			return "P010 BT.601 (full range)";
		case FAST_SDI_P010_BT601:
			return "P010 BT.601";
		case FAST_SDI_P010_BT709:
			return "P010 BT.709";
		case FAST_SDI_P010_BT2020:
			return "P010 BT.2020";
		
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601_FR:
			return "YCbCr420 BT.601 (10 bit LSBZ, full range)";
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601:
			return "YCbCr420 BT.601 (10 bit LSBZ)";
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT709:
			return "YCbCr420 BT.709 (10 bit LSBZ)";
		case FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT2020:
			return "YCbCr420 BT.2020 (10 bit LSBZ)";
		
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601_FR:
			return "YCbCr420 BT.601 (10 bit MSBZ, full range)";
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601:
			return "YCbCr420 BT.601 (10 bit MSBZ)";
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT709:
			return "YCbCr420 BT.709 (10 bit MSBZ)";
		case FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT2020:
			return "YCbCr420 BT.2020 (10 bit MSBZ)";

		case FAST_SDI_422_8_YCbCr_PLANAR_BT601_FR:
			return "YCbCr422 BT.601 (full range)";
		case FAST_SDI_422_8_YCbCr_PLANAR_BT601:
			return "YCbCr422 BT.601";
		case FAST_SDI_422_8_YCbCr_PLANAR_BT709:
			return "YCbCr422 BT.709";
		case FAST_SDI_422_8_YCbCr_PLANAR_BT2020:
			return "YCbCr422 BT.2020";

		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601_FR:
			return "YCbCr422 BT.601 (10 bit LSBZ, full range)";
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601:
			return "YCbCr422 BT.601 (10 bit LSBZ)";
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT709:
			return "YCbCr422 BT.709 (10 bit LSBZ)";
		case FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT2020:
			return "YCbCr422 BT.2020 (10 bit LSBZ)";

		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601_FR:
			return "YCbCr422 BT.601 (10 bit MSBZ, full range)";
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601:
			return "YCbCr422 BT.601 (10 bit MSBZ)";
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT709:
			return "YCbCr422 BT.709 (10 bit MSBZ)";
		case FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT2020:
			return "YCbCr422 BT.2020 (10 bit MSBZ)";

		case FAST_SDI_422_8_CbYCrY_BT601_FR:
			return "CbYCrY422 BT.601 (full range)";
		case FAST_SDI_422_8_CbYCrY_BT601:
			return "CbYCrY422 BT.601";
		case FAST_SDI_422_8_CbYCrY_BT709:
			return "CbYCrY422 BT.709";
		case FAST_SDI_422_8_CbYCrY_BT2020:
			return "CbYCrY422 BT.2020";
		
		case FAST_SDI_422_8_CrYCbY_BT601_FR:
			return "CrYCbY422 BT.601 (full range)";
		case FAST_SDI_422_8_CrYCbY_BT601:
			return "CrYCbY422 BT.601";
		case FAST_SDI_422_8_CrYCbY_BT709:
			return "CrYCbY422 BT.709";
		case FAST_SDI_422_8_CrYCbY_BT2020:
			return "CrYCbY422 BT.2020";

		case FAST_SDI_422_10_CbYCrY_PACKED_BT601_FR:
			return "CrYCbY422 BT.601 (10 bit, packed, full range)";
		case FAST_SDI_422_10_CbYCrY_PACKED_BT601:
			return "CrYCbY422 (10 bit, packed, BT.601)";
		case FAST_SDI_422_10_CbYCrY_PACKED_BT709:
			return "CrYCbY422 (10 bit, packed, BT.709)";
		case FAST_SDI_422_10_CbYCrY_PACKED_BT2020:
			return "CrYCbY422 (10 bit, packed, BT.2020)";
		
		case FAST_SDI_444_8_YCbCr_PLANAR_BT601_FR:
			return "YCbCr444 BT.601 (full range)";
		case FAST_SDI_444_8_YCbCr_PLANAR_BT601:
			return "YCbCr444 BT.601";
		case FAST_SDI_444_8_YCbCr_PLANAR_BT709:
			return "YCbCr444 BT.709";
		case FAST_SDI_444_8_YCbCr_PLANAR_BT2020:
			return "YCbCr444 BT.2020";
		
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601_FR:
			return "YCbCr444 BT.601 (10 bit LSBZ, full range)";
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601:
			return "YCbCr444 BT.601 (10 bit LSBZ)";
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT709:
			return "YCbCr444 BT.709 (10 bit LSBZ)";
		case FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT2020:
			return "YCbCr444 BT.2020 (10 bit LSBZ)";

		case FAST_SDI_RGBA:
			return "RGBA (Alpha)";

		case FAST_SDI_RGB_10_BMR10L:
			return "BMR (10 bits)";
		case FAST_SDI_RGB_10_BMR10B:
			return "BMR (10 bits, big endian)";
		case FAST_SDI_RGB_12_BMR12B:
			return "BMR (12 bits, big endian)";
		case FAST_SDI_RGB_12_BMR12L:
			return "BMR (12 bits)";
		default:
			return "Other";
	}
}

const char* EnumToString(fastColorSaturationOperationType_t value) {
	switch (value) {
		case FAST_OP_REPLACE:
			return "FAST_OP_REPLACE";
		case FAST_OP_MULTIPLY:
			return "FAST_OP_MULTIPLY";
		case FAST_OP_ADD:
			return "FAST_OP_ADD";
	}
	return "Other";
}

const char* EnumToString(fastJpegFormat_t value) {
	switch (value) {
		case FAST_JPEG_Y:
			return "grayscale";
		case FAST_JPEG_444:
			return "444";
		case FAST_JPEG_422:
			return "422";
		case FAST_JPEG_420:
			return "420";
		default:
			return "Other";
	}
}

const char* EnumToString(fastHistogramType_t value) {
	switch (value) {
		case FAST_HISTOGRAM_COMMON:
			return "FAST_HISTOGRAM_COMMON";
		case FAST_HISTOGRAM_BAYER:
			return "FAST_HISTOGRAM_BAYER";
		case FAST_HISTOGRAM_BAYER_G1G2:
			return "FAST_HISTOGRAM_BAYER_G1G2";
		case FAST_HISTOGRAM_PARADE:
			return "FAST_HISTOGRAM_PARADE";
		default:
			return "Other";
	}
}

const char* EnumToString(fastChannelType_t value) {
	switch (value) {
	case FAST_CHANNEL_R:
		return "R";
	case FAST_CHANNEL_G:
		return "G";
	case FAST_CHANNEL_B:
		return "B";
	default:
		return "Other";
	}
}

const char* EnumToString(fastResizeType_t value) {
	switch (value) {
	case FAST_LANCZOS:
		return "lanczos";
	default:
		return "Other";
	}
}

const char* EnumToString(fastAffineTransformations_t value) {
	switch (value) {
	case FAST_AFFINE_FLIP:
		return "flip";
	case FAST_AFFINE_FLOP:
		return "flop";
	case FAST_AFFINE_ROTATION180:
		return "rotation 180";
	case FAST_AFFINE_ROTATION90LEFT:
		return "rotation 90 left";
	case FAST_AFFINE_ROTATION90RIGHT:
		return "rotation 90 right";
	default:
		return "Other";
	}
}

const char* EnumToString(fastImageFilterType_t value) {
	switch (value) {
	case FAST_GAUSSIAN_SHARPEN:
		return "Gaussian Sharpen";
	case FAST_LUT_8_8:
		return "LUT 8->8";
	case FAST_LUT_8_8_C:
		return "LUT 8->8 color";
	case FAST_LUT_8_12:
		return "LUT 8->12";
	case FAST_LUT_8_12_C:
		return "LUT 8->12 color";
	case FAST_LUT_8_16:
		return "LUT 8->16";
	case FAST_LUT_8_16_C:
		return "LUT 8->16 color";
	case FAST_LUT_8_16_BAYER:
		return "LUT 8->16 bayer";

	case FAST_LUT_10_16:
		return "LUT 10->16";
	case FAST_LUT_10_16_BAYER:
		return "LUT 10->16 bayer";

	case FAST_LUT_12_8:
		return "LUT 12->8";
	case FAST_LUT_12_8_C:
		return "LUT 12->8 color";
	case FAST_LUT_12_12:
		return "LUT 12->12";
	case FAST_LUT_12_12_C:
		return "LUT 12->12 color";
	case FAST_LUT_12_16:
		return "LUT 12->16";
	case FAST_LUT_12_16_C:
		return "LUT 12->16 color";
	case FAST_LUT_12_16_BAYER:
		return "LUT 12->16 bayer";

	case FAST_LUT_14_16:
		return "LUT 14->16";
	case FAST_LUT_14_16_BAYER:
		return "LUT 14->16 bayer";

	case FAST_LUT_16_16:
		return "LUT 16->16";
	case FAST_LUT_16_16_C:
		return "LUT 16->16 color";
	case FAST_LUT_16_8:
		return "LUT 16->8";
	case FAST_LUT_16_8_C:
		return "LUT 16->8 color";
	case FAST_LUT_16_16_BAYER:
		return "LUT 16->16 bayer";

	case FAST_LUT_16_16_FR:
		return "LUT 16->16 (full range)";
	case FAST_LUT_16_16_FR_C:
		return "LUT 16->16 (full range, color)";
	case FAST_LUT_16_16_FR_BAYER:
		return "LUT 16->16 (full range, bayer)";

	case FAST_HSV_LUT_3D:
		return "HSV 3D LUT";
	case FAST_RGB_LUT_3D:
		return "RGB 3D LUT";

	case FAST_TONE_CURVE:
		return "Tone Curve";

	case FAST_SAM:
		return "SAM";
	case FAST_SAM16:
		return "SAM16";

	case FAST_BASE_COLOR_CORRECTION:
		return "base color correction";
	case FAST_WHITE_BALANCE:
		return "white balance";
	case FAST_COLUMN_FILTER:
		return "column filter";
	case FAST_COLOR_SATURATION_HSL:
		return "color saturation HSL";
	case FAST_COLOR_SATURATION_HSV:
		return "color saturation HSV";

	case FAST_MEDIAN:
		return "Median filter";
	case FAST_BAYER_BLACK_SHIFT:
		return "Bayer black shift";
	case FAST_DEFRINGE:
		return "Defringe";
	case FAST_BAD_PIXEL_CORRECTION_5X5:
		return "Bad Pixel Correction (5x5)";
	case FAST_BINNING:
		return "Binning";

	default:
		return "Other";
	}
}

const char* EnumToString(fastRawFormat_t value) {
	switch (value) {
	case FAST_RAW_XIMEA12:
		return "XIMEA 12";
	case FAST_RAW_PTG12:
		return "POINT GRAY 12";
	default:
		return "Other";
	}
}

const char* EnumToString(fastDebayerType_t value) {
	switch (value) {
	case FAST_DFPD:
		return "DFPD";
	case FAST_HQLI:
		return "HQLI";
	case FAST_L7:
		return "L7";
	case FAST_MG:
		return "MG";
	case FAST_AMAZE:
		return "AMAZE";
	default:
		return "Other";
	}
}

const char* EnumToString(fastStatus_t value) {
	switch (value) {
	case FAST_OK:
		return "FAST_OK";
	case FAST_TRIAL_PERIOD_EXPIRED:
		return "FAST_TRIAL_PERIOD_EXPIRED";
	case FAST_INVALID_DEVICE:
		return "FAST_INVALID_DEVICE";
	case FAST_INCOMPATIBLE_DEVICE:
		return "FAST_INCOMPATIBLE_DEVICE";

	case FAST_INSUFFICIENT_DEVICE_MEMORY:
		return "FAST_INSUFFICIENT_DEVICE_MEMORY";
	case FAST_INSUFFICIENT_HOST_MEMORY:
		return "FAST_INSUFFICIENT_HOST_MEMORY";
	case FAST_INVALID_HANDLE:
		return "FAST_INVALID_HANDLE";
	case FAST_INVALID_VALUE:
		return "FAST_INVALID_VALUE";
	case FAST_UNAPPLICABLE_OPERATION:
		return "FAST_UNAPPLICABLE_OPERATION";
	case FAST_INVALID_SIZE:
		return "FAST_INVALID_SIZE";

	case FAST_UNALIGNED_DATA:
		return "FAST_UNALIGNED_DATA";
	case FAST_INVALID_TABLE:
		return "FAST_INVALID_TABLE";
	case FAST_BITSTREAM_CORRUPT:
		return "FAST_BITSTREAM_CORRUPT";
	case FAST_EXECUTION_FAILURE:
		return "FAST_EXECUTION_FAILURE";
	case FAST_INTERNAL_ERROR:
		return "FAST_INTERNAL_ERROR";

	case FAST_UNSUPPORTED_SURFACE:
		return "FAST_UNSUPPORTED_SURFACE";
	case FAST_IO_ERROR:
		return "FAST_IO_ERROR";
	case FAST_INVALID_FORMAT:
		return "FAST_INVALID_FORMAT";
	case FAST_UNSUPPORTED_FORMAT:
		return "FAST_UNSUPPORTED_FORMAT";

	case FAST_MJPEG_THREAD_ERROR:
		return "FAST_UNSUPPORTED_SURFACE";
	case FAST_TIMEOUT:
		return "FAST_TIMEOUT";
	case FAST_MJPEG_OPEN_FILE_ERROR:
		return "FAST_INVALID_FORMAT";
	case FAST_UNKNOWN_ERROR:
		return "FAST_UNSUPPORTED_FORMAT";

	default:
		return "other";
	}
}

const char* EnumToString(fastSurfaceFormat_t value) {
	switch (value) {
	case FAST_I8:
		return "I8";
	case FAST_I10:
		return "I10";
	case FAST_I12:
		return "I12";
	case FAST_I14:
		return "I14";

	case FAST_RGB8:
		return "RGB8";
	case FAST_BGR8:
		return "BGR8";
	case FAST_RGB12:
		return "RGB12";
	case FAST_RGB16:
		return "RGB16";

	case FAST_CrCbY8:
		return "CrCbY";
	case FAST_YCbCr8:
		return "YCbCr";
	default:
		return "Other";
	}
}

const char* EnumToString(fastBayerPattern_t value) {
	switch (value) {
	case FAST_BAYER_NONE:
		return "none";
	case FAST_BAYER_RGGB:
		return "RGGB";
	case FAST_BAYER_BGGR:
		return "BGGR";
	case FAST_BAYER_GBRG:
		return "GBRG";
	case FAST_BAYER_GRBG:
		return "GRBG";
	default:
		return "Other";
	}
}

const char* EnumToString(fastBinningMode_t value) {
	switch (value) {
	case FAST_BINNING_NONE:
		return "none";
	case FAST_BINNING_SUM:
		return "sum";
	case FAST_BINNING_AVERAGE:
		return "average";
	default:
		return "Other";
	}
}

const char* EnumToString(fastConvertType_t value) {
	switch (value) {
	case FAST_CONVERT_NONE:
		return "none";
	case FAST_CONVERT_BGR:
		return "BGR";
	default:
		return "Other";
	}
}

const char * EnumToString( fastLicenseType_t value) {
	switch(value) {
		case FAST_LICENSE_TYPE_TRIAL:
			return "Trial";
		case FAST_LICENSE_TYPE_DEMO:
			return "Demo";
		case FAST_LICENSE_TYPE_STANDARD_SENSELOCK:
			return "Standard Senselock Provider";
		case FAST_LICENSE_TYPE_STANDARD_GUARDANT:
			return "Standard Guardant Provider";
		case FAST_LICENSE_TYPE_ENTERPRISE:
			return "Enterprise";
	}

	return "Other";
}

const char* EnumToString(fastLicenseProvider_t value) {
	switch (value) {
		case FAST_LICENSE_PROVIDER_NONE:
			return "none";
		case FAST_LICENSE_PROVIDER_SENSELOCK_DONGLE:
			return "Senselock Dongle";
		case FAST_LICENSE_PROVIDER_GUARDANT_DONGLE:
			return "Guardant Dongle";
		case FAST_LICENSE_PROVIDER_GUARDANT_SOFT_KEY:
			return "Guardant Soft Key";
		}
	return "Other";
}
