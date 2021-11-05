/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FAST_H__
#define __FAST_H__

#include <stdlib.h>

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
// Basic data types
////////////////////////////////////////////////////////////////////////////////
#define FAST_ALIGNMENT 4U
#define FAST_SCALE_FACTOR_MAX 40U
#define FAST_MIN_SCALED_SIZE 32U

typedef enum {
	FAST_OK,                           // There is no error during function execution

	FAST_TRIAL_PERIOD_EXPIRED,

	FAST_INVALID_DEVICE,               // Device with selected index does not exist or device is non NVIDIA device or device is non CUDA-compatible device
	FAST_INCOMPATIBLE_DEVICE,          // Device is CUDA-compatible, but its compute compatibility is below 2.0, thus device is considered to be incompatible with SDK
	FAST_INSUFFICIENT_DEVICE_MEMORY,   // Available device memory is not enough to allocate new buffer
	FAST_INSUFFICIENT_HOST_MEMORY,     // Available host memory is not enough to allocate new buffer
	FAST_INVALID_HANDLE,               // Component handle is invalid or has inappropriate type
	FAST_INVALID_VALUE,                // Some parameter of the function called is invalid or combination of input parameters are unacceptable
	FAST_UNAPPLICABLE_OPERATION,	   // This operation can not be applied to the current type of data
	FAST_INVALID_SIZE,                 // Image dimension is invalid
	FAST_UNALIGNED_DATA,               // Buffer base pointers or pitch are not properly aligned
	FAST_INVALID_TABLE,                // Invalid quantization / Huffman table
	FAST_BITSTREAM_CORRUPT,            // JPEG bitstream is corrupted and can not be decoded
	FAST_EXECUTION_FAILURE,            // Device kernel execution failure
	FAST_INTERNAL_ERROR,               // Internal error, non-kernel software execution failure
	FAST_UNSUPPORTED_SURFACE,

	FAST_IO_ERROR,                     // Failed to read/write file
	FAST_INVALID_FORMAT,               // Invalid file format
	FAST_UNSUPPORTED_FORMAT,           // File format is not supported by the current version of SDK
	FAST_END_OF_STREAM,

	FAST_MJPEG_THREAD_ERROR,
	FAST_TIMEOUT,
	FAST_MJPEG_OPEN_FILE_ERROR,

	FAST_UNKNOWN_ERROR,                 // Unrecognized error
	FAST_INCOMPATIBLE_DRIVER
} fastStatus_t;

typedef enum {
	FAST_I8,
	FAST_I10,
	FAST_I12,
	FAST_I14,
	FAST_I16,

	FAST_RGB8,
	FAST_BGR8,
	FAST_RGB12,
	FAST_RGB16,

	FAST_BGRX8,

	FAST_CrCbY8,
	FAST_YCbCr8
} fastSurfaceFormat_t;

typedef enum {
	FAST_JPEG_SEQUENTIAL_DCT, FAST_JPEG_LOSSLESS
} fastJpegMode_t;

typedef enum {
	FAST_JPEG_Y, FAST_JPEG_444, FAST_JPEG_422, FAST_JPEG_420
} fastJpegFormat_t;

typedef enum {
	FAST_BAYER_NONE,
	FAST_BAYER_RGGB,
	FAST_BAYER_BGGR,
	FAST_BAYER_GBRG,
	FAST_BAYER_GRBG,
} fastBayerPattern_t;

typedef enum {
	FAST_DFPD,
	FAST_HQLI,
	FAST_MG,
	FAST_MG_V2,

	FAST_BINNING_2x2,
	FAST_BINNING_4x4,
	FAST_BINNING_8x8,
	FAST_L7,
	FAST_AMAZE
} fastDebayerType_t;

typedef enum {
	FAST_RAW_XIMEA12,
	FAST_RAW_PTG12,
} fastRawFormat_t;

typedef enum {
	FAST_SDI_YV12_BT601_FR,
	FAST_SDI_YV12_BT601,
	FAST_SDI_YV12_BT709,
	FAST_SDI_YV12_BT2020,

	FAST_SDI_NV12_BT601_FR,
	FAST_SDI_NV12_BT601,
	FAST_SDI_NV12_BT709,
	FAST_SDI_NV12_BT2020,

	FAST_SDI_P010_BT601_FR,
	FAST_SDI_P010_BT601,
	FAST_SDI_P010_BT709,
	FAST_SDI_P010_BT2020,

	FAST_SDI_420_8_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_420_8_YCbCr_PLANAR_BT601,
	FAST_SDI_420_8_YCbCr_PLANAR_BT709,
	FAST_SDI_420_8_YCbCr_PLANAR_BT2020,

	FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601,
	FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT709,
	FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT2020,

	FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601,
	FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT709,
	FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT2020,

	FAST_SDI_422_8_CbYCrY_BT601_FR,
	FAST_SDI_422_8_CbYCrY_BT601,
	FAST_SDI_422_8_CbYCrY_BT709,
	FAST_SDI_422_8_CbYCrY_BT2020,

	FAST_SDI_422_8_CrYCbY_BT601_FR,
	FAST_SDI_422_8_CrYCbY_BT601,
	FAST_SDI_422_8_CrYCbY_BT709,
	FAST_SDI_422_8_CrYCbY_BT2020,

	FAST_SDI_422_10_CbYCrY_PACKED_BT2020,
	FAST_SDI_422_10_CbYCrY_PACKED_BT601_FR,
	FAST_SDI_422_10_CbYCrY_PACKED_BT601,
	FAST_SDI_422_10_CbYCrY_PACKED_BT709,

	FAST_SDI_422_8_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_422_8_YCbCr_PLANAR_BT601,
	FAST_SDI_422_8_YCbCr_PLANAR_BT709,
	FAST_SDI_422_8_YCbCr_PLANAR_BT2020,

	FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601,
	FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT709,
	FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT2020,

	FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601,
	FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT709,
	FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT2020,

	FAST_SDI_444_8_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_444_8_YCbCr_PLANAR_BT601,
	FAST_SDI_444_8_YCbCr_PLANAR_BT709,
	FAST_SDI_444_8_YCbCr_PLANAR_BT2020,

	FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601,
	FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT709,
	FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT2020,

	FAST_SDI_RGBA,

	FAST_SDI_RGB_10_BMR10L,
	FAST_SDI_RGB_12_BMR12B,
	FAST_SDI_RGB_12_BMR12L,
	FAST_SDI_RGB_10_BMR10B

} fastSDIFormat_t;


typedef enum {
	FAST_RGBA_ALPHA_PADDING_ZERO,
	FAST_RGBA_ALPHA_PADDING_FF,
} fastRGBAAlphaPadding_t;

typedef enum {
	FAST_CHANNEL_H,
	FAST_CHANNEL_S,
	FAST_CHANNEL_L_OR_V
} fastColorSaturationChannelType_t;

typedef enum {
	FAST_CHANNEL_R,
	FAST_CHANNEL_G,
	FAST_CHANNEL_B
} fastChannelType_t;

typedef enum {
	FAST_OP_REPLACE,
	FAST_OP_ADD,
	FAST_OP_MULTIPLY
} fastColorSaturationOperationType_t;

typedef enum {
	FAST_BINNING_NONE,
	FAST_BINNING_SUM,
	FAST_BINNING_AVERAGE
} fastBinningMode_t;

typedef enum {
	FAST_GAUSSIAN_SHARPEN = 1,
	FAST_LUT_8_8,
	FAST_LUT_8_8_C,
	FAST_LUT_8_12,
	FAST_LUT_8_12_C,
	FAST_LUT_8_16,
	FAST_LUT_8_16_C,
	FAST_LUT_8_16_BAYER,

	FAST_LUT_10_16,
	FAST_LUT_10_16_BAYER,

	FAST_LUT_12_8,
	FAST_LUT_12_8_C,
	FAST_LUT_12_12,
	FAST_LUT_12_12_C,
	FAST_LUT_12_16,
	FAST_LUT_12_16_C,
	FAST_LUT_12_16_BAYER,

	FAST_LUT_14_16,
	FAST_LUT_14_16_BAYER,

	FAST_LUT_16_16,
	FAST_LUT_16_16_C,
	FAST_LUT_16_8,
	FAST_LUT_16_8_C,
	FAST_LUT_16_16_BAYER,

	FAST_LUT_16_16_FR,
	FAST_LUT_16_16_FR_C,
	FAST_LUT_16_16_FR_BAYER,

	FAST_HSV_LUT_3D,
	FAST_RGB_LUT_3D,

	FAST_TONE_CURVE,
	
	FAST_SAM,
	FAST_SAM16,

	FAST_BASE_COLOR_CORRECTION,
	FAST_WHITE_BALANCE,
	FAST_COLUMN_FILTER,
	FAST_COLOR_SATURATION_HSL,
	FAST_COLOR_SATURATION_HSV,

	FAST_MEDIAN,
	FAST_BAYER_BLACK_SHIFT,
	FAST_DEFRINGE,
	FAST_BAD_PIXEL_CORRECTION_5X5,
	FAST_BINNING,
	FAST_GAUSSIAN_BLUR
} fastImageFilterType_t;

typedef enum {
	FAST_BIT_DEPTH,
	FAST_SELECT_CHANNEL,
	FAST_RGB_TO_GRAYSCALE,
	FAST_GRAYSCALE_TO_RGB,
	FAST_BAYER_TO_RGB
} fastSurfaceConverter_t;

typedef enum {
	FAST_AFFINE_NOP = 0,
	FAST_AFFINE_FLIP = 1,
	FAST_AFFINE_FLOP = 2,
	FAST_AFFINE_ROTATION180 = 4,
	FAST_AFFINE_ROTATION90LEFT = 8,
	FAST_AFFINE_ROTATION90LEFT_FLOPPED = 16,
	FAST_AFFINE_ROTATION90RIGHT = 32,
	FAST_AFFINE_ROTATION90RIGHT_FLOPPED = 64,
	FAST_AFFINE_ALL = 127
} fastAffineTransformations_t;

typedef enum {
	FAST_LANCZOS = 1
} fastResizeType_t;

typedef enum {
	FAST_CONVERT_NONE,
	FAST_CONVERT_BGR
} fastConvertType_t;

typedef enum {
	FAST_HISTOGRAM_COMMON,
	FAST_HISTOGRAM_BAYER,
	FAST_HISTOGRAM_BAYER_G1G2,
	FAST_HISTOGRAM_PARADE,
} fastHistogramType_t;

typedef enum {
	FAST_LICENSE_TYPE_DEMO,
	FAST_LICENSE_TYPE_TRIAL,
	FAST_LICENSE_TYPE_STANDARD_SENSELOCK,
	FAST_LICENSE_TYPE_STANDARD_GUARDANT,
	FAST_LICENSE_TYPE_ENTERPRISE
} fastLicenseType_t;

typedef enum {
	FAST_LICENSE_PROVIDER_NONE,
	FAST_LICENSE_PROVIDER_SENSELOCK_DONGLE,
	FAST_LICENSE_PROVIDER_GUARDANT_DONGLE,
	FAST_LICENSE_PROVIDER_GUARDANT_SOFT_KEY
} fastLicenseProvider_t;


////////////////////////////////////////////////////////////////////////////////
// 
////////////////////////////////////////////////////////////////////////////////
typedef struct fastEncoderHandleStruct *fastJpegEncoderHandle_t;
typedef struct fastDecoderHandleStruct *fastJpegDecoderHandle_t;

typedef struct fastEncoderHandleStruct_v2 *fastJpegEncoderHandle_v2_t;
typedef struct fastDecoderHandleStruct_v2 *fastJpegDecoderHandle_v2_t;

typedef struct fastDebayerHandleStruct *fastDebayerHandle_t;
typedef struct fastResizerHandleStruct *fastResizerHandle_t;
typedef struct fastSurfaceConverterHandleStruct *fastSurfaceConverterHandle_t;
typedef struct fastImageFiltersHandleStruct *fastImageFiltersHandle_t;
typedef struct fastCropHandleStruct *fastCropHandle_t;
typedef struct fastGpuTimerStruct *fastGpuTimerHandle_t;
typedef struct fastExportToHostStruct *fastExportToHostHandle_t;
typedef struct fastExportToDeviceStruct *fastExportToDeviceHandle_t;
typedef struct fastImportFromHostStruct *fastImportFromHostHandle_t;
typedef struct fastImportFromDeviceStruct *fastImportFromDeviceHandle_t;
typedef struct fastAffineHandleStruct *fastAffineHandle_t;
typedef struct fastRawUnpackerHandleStruct *fastRawUnpackerHandle_t;
typedef struct fastBayerMergerHandleStruct *fastBayerMergerHandle_t;
typedef struct fastBayerSplitterHandleStruct *fastBayerSplitterHandle_t;
typedef struct fastMuxStruct *fastMuxHandle_t;
typedef struct fastSDIImportFromHostHandleStruct *fastSDIImportFromHostHandle_t;
typedef struct fastSDIImportFromDeviceHandleStruct *fastSDIImportFromDeviceHandle_t;
typedef struct fastSDIExportToHostHandleStruct *fastSDIExportToHostHandle_t;
typedef struct fastSDIExportToDeviceHandleStruct *fastSDIExportToDeviceHandle_t;
typedef struct fastHistogramHandleStruct *fastHistogramHandle_t;

typedef struct fastSdkParameters *fastSdkParametersHandle_t;

typedef struct fastDeviceSurfaceBuffer_t *fastDeviceSurfaceBufferHandle_t;

typedef struct {
	unsigned short R;
	unsigned short G;
	unsigned short B;
} fastRgb_t;

typedef struct {
	float H;
	float S;
	float V;
} fastHSVfloat_t;

typedef struct{
	unsigned short data[64];
} fastQuantTable_t;

typedef struct {
	fastQuantTable_t table[4];
} fastJpegQuantState_t;

typedef struct {
	unsigned char bucket[16];
	unsigned char alphabet[256];
} fastHuffmanTable_t;

typedef struct {
	fastHuffmanTable_t table[2][2];
} fastJpegHuffmanState_t;

typedef struct {
	unsigned quantTableMask;
	unsigned huffmanTableMask[2];
	unsigned scanChannelMask;
	unsigned scanGroupMask;
} fastJpegScanStruct_t;

typedef struct {
	unsigned short exifCode;
	char *exifData;
	int exifLength;
} fastJpegExifSection_t;

typedef struct {
	fastJpegMode_t jpegMode;
	fastJpegFormat_t jpegFmt;

	int predictorClass;

	unsigned char *h_Bytestream;
	unsigned bytestreamSize;
	unsigned headerSize;

	unsigned height;
	unsigned width;
	unsigned bitsPerChannel;

	fastJpegExifSection_t *exifSections;
	unsigned exifSectionsCount;

	fastJpegQuantState_t quantState;
	fastJpegHuffmanState_t huffmanState;
	fastJpegScanStruct_t scanMap;
	unsigned restartInterval;
} fastJfifInfo_t;

typedef struct {
	fastJpegMode_t jpegMode;
	fastJpegFormat_t jpegFmt;

	unsigned char *d_Bytestream;
	unsigned bytestreamSize;

	unsigned height;
	unsigned width;
	unsigned bitsPerChannel;

	fastJpegExifSection_t *exifSections;
	unsigned exifSectionsCount;

	fastJpegQuantState_t quantState;
	fastJpegHuffmanState_t huffmanState;
	fastJpegScanStruct_t scanMap;
	unsigned restartInterval;
} fastJfifInfoAsync_t;

typedef struct {
	double sigma;
} fastGaussianFilter_t;

typedef struct {
	int columnOffset;
} fastColumnFilter_t;

typedef struct {
	unsigned char lut[256];
} fastLut_8_t;

typedef struct {
	unsigned char lut_R[256];
	unsigned char lut_G[256];
	unsigned char lut_B[256];
} fastLut_8_C_t;

typedef struct {
	unsigned short lut[256];
} fastLut_8_16_t;

typedef struct {
	unsigned short lut_R[256];
	unsigned short lut_G[256];
	unsigned short lut_B[256];
} fastLut_8_16_C_t;

typedef struct {
	unsigned short lut_R[256];
	unsigned short lut_G[256];
	unsigned short lut_B[256];

	fastBayerPattern_t pattern;
} fastLut_8_16_Bayer_t;

typedef struct {
	unsigned short lut[1024];
} fastLut_10_t;

typedef struct {
	unsigned short lut_R[1024];
	unsigned short lut_G[1024];
	unsigned short lut_B[1024];

	fastBayerPattern_t pattern;
} fastLut_10_16_Bayer_t;

typedef struct {
	unsigned char lut[4096];
} fastLut_12_8_t;

typedef struct {
	unsigned char lut_R[4096];
	unsigned char lut_G[4096];
	unsigned char lut_B[4096];
} fastLut_12_8_C_t;

typedef struct {
	unsigned short lut_R[4096];
	unsigned short lut_G[4096];
	unsigned short lut_B[4096];

	fastBayerPattern_t pattern;
} fastLut_12_16_Bayer_t;

typedef struct {
	unsigned short lut[4096];
} fastLut_12_t;

typedef struct {
	unsigned short lut_R[4096];
	unsigned short lut_G[4096];
	unsigned short lut_B[4096];
} fastLut_12_C_t;

typedef struct {
	unsigned short lut_R[16384];
	unsigned short lut_G[16384];
	unsigned short lut_B[16384];

	fastBayerPattern_t pattern;
} fastLut_14_16_Bayer_t;

typedef struct {
	unsigned short lut[16384];
} fastLut_16_t;

typedef struct {
	unsigned short lut_R[16384];
	unsigned short lut_G[16384];
	unsigned short lut_B[16384];
} fastLut_16_C_t;

typedef struct {
	unsigned short lut_R[16384];
	unsigned short lut_G[16384];
	unsigned short lut_B[16384];

	fastBayerPattern_t pattern;
} fastLut_16_Bayer_t;

typedef struct {
	unsigned short lut[65536];
} fastLut_16_FR_t;

typedef struct {
	unsigned short lut_R[65536];
	unsigned short lut_G[65536];
	unsigned short lut_B[65536];
} fastLut_16_FR_C_t;

typedef struct {
	unsigned short lut_R[65536];
	unsigned short lut_G[65536];
	unsigned short lut_B[65536];

	fastBayerPattern_t pattern;
} fastLut_16_FR_Bayer_t;

typedef struct {
	unsigned char lut[16384];
} fastLut_16_8_t;

typedef struct {
	unsigned char lut_R[16384];
	unsigned char lut_G[16384];
	unsigned char lut_B[16384];
} fastLut_16_8_C_t;

typedef struct {
	float *lutR;
	float *lutG;
	float *lutB;

	unsigned size1D;
} fastRGBLut_3D_t;

typedef struct {
	unsigned int dimH;
	unsigned int dimS;
	unsigned int dimV;

	fastColorSaturationOperationType_t operationH;
	fastColorSaturationOperationType_t operationS;
	fastColorSaturationOperationType_t operationV;

	float *LutH;
	float *LutS;
	float *LutV;
} fastHsvLut3D_t;

typedef struct {
	char *blackShiftMatrix;
	float *correctionMatrix;
} fastSam_t;

typedef struct {
	short *blackShiftMatrix;
	float *correctionMatrix;
} fastSam16_t;

typedef struct {
	fastBinningMode_t mode;
	unsigned factorX;
	unsigned factorY;
} fastBinning_t;

typedef struct {
	float matrix[12];
	int whiteLevel[3];
} fastBaseColorCorrection_t;

typedef struct {
	float Lut[3][1024];

	fastColorSaturationOperationType_t operation[3];
	fastColorSaturationChannelType_t sourceChannel[3];
} fastColorSaturation_t;

typedef struct {
	unsigned short toneCurve[1024];
} fastToneCurve_t;

typedef struct {
	fastBayerPattern_t pattern;
} fastBadPixelCorrection_t;

typedef struct {
	unsigned windowSize;
	unsigned short tint[3]; /*RGB*/
	float fi_tint;

	float fi_max;
	float coefficient;
} fastDefringe_t;

typedef struct {
	fastConvertType_t convert;
} fastExportParameters_t;

typedef struct {
	float R;
	float G1;
	float G2;
	float B;

	fastBayerPattern_t bayerPattern;
} fastWhiteBalance_t;

typedef struct {
	float R;
	float G;
	float B;

	fastBayerPattern_t bayerPattern;
} fastBayerBlackShift_t;

typedef struct {
	unsigned overrideSourceBitsPerChannel;
	bool isOverrideSourceBitsPerChannel;
	unsigned targetBitsPerChannel;
} fastBitDepthConverter_t;

typedef struct {
	fastChannelType_t channel;
} fastSelectChannel_t;

typedef struct {
	float coefficientR;
	float coefficientG;
	float coefficientB;
} fastRgbToGrayscale_t;

typedef struct {
	unsigned char *data;
	unsigned width;
	unsigned pitch;
	unsigned height;
} fastChannelDescription_t;

typedef struct {
	fastBayerPattern_t bayerPattern;
} fastBayerPatternParam_t;

typedef struct {
	unsigned int stride;
} fastHistogramParade_t;

typedef struct {
	fastRGBAAlphaPadding_t padding;
} fastSDIRGBAExport_t;

typedef struct {
	unsigned overrideSourceBitsPerChannel;
} fastSDIYCbCrExport_t;

typedef struct {
	bool isConvert12to16;
} fastSDIRaw12Import_t;

typedef struct {
	unsigned short sdkLicenseVersion;
	unsigned short dongleLicenseVersion;
	char dongleName[56];
	char dongleId[8];
} fastLicenseProviderSenselockInfo_t;

typedef struct {
	int id;
} fastLicenseProviderGuardantFeature_t;

typedef struct {
	int id;
	int featuresCount;
	fastLicenseProviderGuardantFeature_t features[4];
} fastLicenseProviderGuardantProduct_t;


typedef struct {
	unsigned int dongleId;
	unsigned int productsCount;
	fastLicenseProviderGuardantProduct_t products[4];
} fastLicenseProviderGuardantInfo_t;

typedef struct {
	fastLicenseType_t licenseType;
	unsigned char sdkVersion[4];
	char buildDate[11];
	int remainingTrialDays;
	fastLicenseProvider_t licenseProvider;
	union {
		fastLicenseProviderSenselockInfo_t senselockInfo;
		fastLicenseProviderGuardantInfo_t guardantInfo;
	};
} fastLicenseInfo_t;

typedef struct {
	unsigned surfaceFmt;

	unsigned width;
	unsigned height;
	unsigned pitch;

	unsigned maxWidth;
	unsigned maxHeight;
	unsigned maxPitch;
} fastDeviceSurfaceBufferInfo_t;

typedef struct {
	char name[256];
	int major;
	int minor;
	int integrated;
	int isMultiGpuBoard;
	int	pciDeviceID;
	size_t totalMem;
} fastDeviceProperty;


///////////////////////////////////////////////////////////////////////////////
// Master SDK and secondary library initialization
///////////////////////////////////////////////////////////////////////////////
/*! \brief Set GPU device to work with.

	Set GPU device to work with.
	If device is not found or device is not NVIDIA device or device does not support CUDA,
	function will return status FAST_INVALID_DEVICE.
	If device has compute compatibility below 2.0, function will return status FAST_INCOMPATIBLE_DEVICE.

	\param [in] affinity   affinity mask. "1" in less significant bit of affinity mask denotes to use GPU with device Id = 1. "1" in next bit denotes to use GPU with device Id = 2 and so on
	\param [in] openGlMode   if openGlMode set to be true, then FASTVIDEO SDK is initialized to work with OpenGL application. In other case FASTVIDEO SDK is initialized to work with ordinary CUDA application
*/
extern fastStatus_t DLL fastInit(unsigned affinity, bool openGlMode);

/*! \brief Get handle of SDK global options.

	Get handle of SDK global options.

	\param [out] handle   pointer to global options handle.
*/
extern fastStatus_t DLL fastGetSdkParametersHandle(fastSdkParametersHandle_t *handle);

////////////////////////////////////////////////////////////////////////////////
// Trace and Auxiliary functions
////////////////////////////////////////////////////////////////////////////////

/*! \brief Set value of interface synchronization in global option.

	Get public information from device surface buffer.
	Structure fastDeviceSurfaceBufferInfo_t contains the following fields:
	surfaceFmt - type of surface. Value is fastSurfaceFormat_t enum integer values. Values 12 and 13 are internal representations of FAST_RGB12 and FAST_RGB16 formats respectively. Value is defined after component creation.
	maxWidth - maximum width of processed image. Value is defined after component creation.
	maxHeight - maximum height of processed image. Value is defined after component creation.
	maxPitch - pitch for maximal image. Value is defined after component creation.
	width - width of currently processed image. Value is defined after call of transform function.
	height - height of currently processed image. Value is defined after call of transform function.
	pitch - pitch of currently processed image. Value is defined after call of transform function.

	\param [in] buffer   handle of device surface buffer.
	\param [out] devBuffer   pointer to structure with public information.
*/
extern fastStatus_t DLL fastGetDeviceSurfaceBufferInfo(
	fastDeviceSurfaceBufferHandle_t buffer,
	fastDeviceSurfaceBufferInfo_t *devBuffer
);

/*! \brief Set value of interface synchronization in global option.

	Set value of interface synchronization in global option.
	Global option Interface Synchronization adds to the end of all interfaces
	method cudaDeviceSyncronize call. This localizes problem in one interface
	method. Asynchronous error can not pass bound between calls.

	\param [in] isEnabled   option value.
*/
extern fastStatus_t DLL fastEnableInterfaceSynchronization(bool isEnabled);


extern fastStatus_t DLL fastEnableKernelTrace(bool isEnabled);
/*! \brief Open trace file.

	Open trace file.
	If file cannot be opened or created function returns FAST_IO_ERROR.

	\param [in] fileName   file path or file name for trace file.
*/
extern fastStatus_t DLL fastTraceCreate(const char *fileName);

/*! \brief Close current trace file.

	Close current trace file.
*/
extern fastStatus_t DLL fastTraceClose(void);

/*! \brief Set value of trace flush in global option.

	Set value of trace flush in global option.
	If trace flush is enabled, every trace write will be stored to file immediately.

	\param [in] enableFlush   option value.
*/
extern fastStatus_t DLL fastTraceEnableFlush(const bool enableFlush);

/*! \brief .

	.

	\param [in] enableLutDump   _
*/
extern fastStatus_t DLL fastTraceEnableLUTDump(const bool enableLutDump);

/*! \brief .

	.

	\param [in] messagePerFile   _
*/
extern fastStatus_t DLL fastTraceEnableMultipleFiles(int messagePerFile);

/*! \brief .

	.

	\param [in] messagePerFile   _
	\param [in] fileCount   _
*/
extern fastStatus_t DLL fastTraceEnableCyclicFiles(int messagePerFile, int fileCount);

///////////////////////////////////////////////////////////////////////////////
// Memory management functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Allocates page-locked memory on CPU.

	Allocates page-locked memory on CPU.
	Page-locked memory cannot be moved from RAM to swap file.
	It increases PCI-Express I/O speed of GPU over conventional memory.
	If system can not allocate page-locked memory, then function will
	return status  FAST_INSUFFICIENT_HOST_MEMORY.

	\param [out] buffer   pointer to allocated memory
	\param [in] size   size of allocated memory in Bytes
*/
extern fastStatus_t DLL fastMalloc(void **buffer, size_t size);

/*! \brief Frees page-locked memory.

	Frees page-locked memory.

	\param [in] buffer   pointer to allocated memory
*/
extern fastStatus_t DLL fastFree(void *buffer);

/*! \brief .

	.

	\param [out] devices   _
	\param [out] count   _
*/
extern fastStatus_t DLL fastGetDevices(fastDeviceProperty **devices, int *count);

///////////////////////////////////////////////////////////////////////////////
// Timer functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates Timer and returns associated handle.
	
	Creates Timer and returns associated handle.
	Allocates necessary buffers in GPU memory. In case GPU does not have enough
	free memory returns FAST_INSUFFICIENT_DEVICE_MEMORY. 

	\param [out]   pointer to created Timer handle
*/
extern fastStatus_t DLL fastGpuTimerCreate(fastGpuTimerHandle_t *handle);

/*! \brief Inserts start event into GPU stream.

	Inserts start event into GPU stream.
	
	\param [in] handle   Timer handle pointer
*/
extern fastStatus_t DLL fastGpuTimerStart(fastGpuTimerHandle_t handle);

/*! \brief Inserts stop event into GPU stream.

	Inserts stop event into GPU stream.
	
	\param [in] handle   Timer handle pointer.
*/
extern fastStatus_t DLL fastGpuTimerStop(fastGpuTimerHandle_t handle);

/*! \brief Synchronizes CPU thread with stop event and calculates time elapsed between start and stop events.

	Synchronizes CPU thread with stop event and calculates
	time elapsed between start and stop events.

	\param [in] handle   Timer handle pointer
	\param [out] elapsed   time elapsed
*/
extern fastStatus_t DLL fastGpuTimerGetTime(
	fastGpuTimerHandle_t handle,
	float *elapsed
);

/*! \brief Destroys Timer handle.

	Procedure frees all device memory.

	\param [in] handle   pointer to Timer handle
*/
extern fastStatus_t DLL fastGpuTimerDestroy(fastGpuTimerHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// JPEG Encoder functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates JPEG Encoder and returns associated handle.

	Creates JPEG Encoder and returns associated handle.
	Function fastJpegEncoderCreate allocates all necessary buffers in GPU memory. So in case GPU does
	not have enough free memory, then fastJpegEncoderCreate returns FAST_INSUFFICIENT_DEVICE_MEMORY.
	Maximum dimensions of the image are set to Encoder during creation. Thus if encoded image exceeds
	the maximum value, then error status FAST_INVALID_SIZE will be returned.
	Gray image (FAST_I8) can be encoded by color encoder (FAST_RGB8). In this case Encoder converts gray
	image to color image by duplicating gray channel to all color channels.
	If component does not support current surface format then the function will return FAST_UNSUPPORTED_SURFACE.  

	\param [out] handle   pointer to created JPEG Encoder handle
	\param [in] maxWidth   maximum image width in pixels
	\param [in] maxHeight   maximum image height in pixels
	\param [in] srcBuffer   linked buffer from previous component
*/
extern fastStatus_t DLL fastJpegEncoderCreate(
    fastJpegEncoderHandle_t *handle,

    unsigned maxWidth,
    unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief .

	.

	\param [in] handle   JPEG Encoder handle
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastJpegEncoderChangeSrcBuffer(
	fastJpegEncoderHandle_t handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Returns requested GPU memory for JPEG Encoder.

	Returns requested GPU memory for JPEG Encoder.
	Function returns requested memory size in Bytes for JPEG Encoder.
	
	\param [in] handle   JPEG Encoder handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastJpegEncoderGetAllocatedGpuMemorySize(
    fastJpegEncoderHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Encodes surface to JPEG and store to host memory.

	Encodes surface to JPEG and store to host memory.
	The procedure takes surface from previous component of the pipeline through input linked
	buffer and encodes it accordingly addition parameters from jfifInfo. JPEG bytestream is
	placed to h_Bytestream buffer in jfifInfo.
	Buffer for JPEG bytestream in jfifInfo has to be allocated before call. Its recommended
	size is surfaceHeight*surfacePitch4. Real JPEG bytestream size is calculated during
	compression and put to bytestreamSize in jfifInfo. If size of h_Bytestream is not enough,
	then procedure returns status FAST_INTERNAL_ERROR.
	Members of jfifInfo exifSectionsCount and exifSections have to be initialized by 0

	\param [in] handle   JPEG Encoder handle
	\param [in] quality   adjusts output JPEG file size and quality. Quality is an integer value from 1 to 100 where 100 means the best quality and maximum file size of compressed image.
	\param [in] jfifInfo   pointer to fastJfifInfo_t struct that contains all necessary information for JPEG encoding. For more detail see JPEG Encoder Description.
*/
extern fastStatus_t DLL fastJpegEncode(
    fastJpegEncoderHandle_t handle,

	unsigned quality,

	fastJfifInfo_t *jfifInfo
);

/*! \brief Encodes surface to JPEG and store to device memory.

	Encodes surface to JPEG and store to device memory.
	The procedure takes surface from previous component of the pipeline through input linked
	buffer and encodes it accordingly addition parameters from jfifInfo. JPEG bytestream is
	placed to d_Bytestream buffer in jfifInfo. Memory for d_Bytestream is allocated by jpeg encoder.
	Members of jfifInfo exifSectionsCount and exifSections have to be initialized by 0

	\param [in] handle   JPEG Encoder handle
	\param [in] quality   adjusts output JPEG file size and quality. Quality is an integer value from 1 to 100 where 100 means the best quality and maximum file size of compressed image.
	\param [in] jfifInfo   pointer to fastJfifInfoAsync_t struct that contains all necessary information for JPEG encoding. For more detail see JPEG Encoder Description.
*/
extern fastStatus_t DLL fastJpegEncodeAsync(
	fastJpegEncoderHandle_t handle,
	unsigned quality,
	fastJfifInfoAsync_t *jfifInfo
);

/*! \brief .

	.

	\param [in] handle   JPEG Encoder handle
	\param [in] quantTable   _
	\param [in] jfifInfo   _
*/
extern fastStatus_t DLL fastJpegEncodeWithQuantTable(
	fastJpegEncoderHandle_t handle,
	fastJpegQuantState_t *quantTable,
	fastJfifInfo_t *jfifInfo
);

/*! \brief .

	.

	\param [in] handle   JPEG Encoder handle
	\param [in] quantTable   _
	\param [in] jfifInfo   _
*/
extern fastStatus_t DLL fastJpegEncodeAsyncWithQuantTable(
	fastJpegEncoderHandle_t handle,
	fastJpegQuantState_t *quantTable,
	fastJfifInfoAsync_t *jfifInfo
);

/*! \brief Destroys JPEG encoder.

	Destroys JPEG encoder.

	\param [in] handle   JPEG encoder handle
*/
extern fastStatus_t DLL fastJpegEncoderDestroy(fastJpegEncoderHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// JPEG Decoder functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates JPEG Decoder and returns associated handle.

	Creates JPEG Decoder and returns associated handle.
	Function fastJpegDecoderCreate allocates all necessary buffers in GPU memory. Thus
	in case GPU does not have enough free memory, then fastJpegDecoderCreate returns
	FAST_INSUFFICIENT_DEVICE_MEMORY.
	Maximum dimensions of the image are set to Decoder during creation. Thus if
	transformation result exceeds the maximum value, then error status
	FAST_INVALID_SIZE will be returned.
	Only FAST_RGB8 and FAST_I8 surface formats are supported in other case
	FAST_UNSUPPORTED_SURFACE will be returned.

	\param [out] handle   pointer to created JPEG Decoder 
	\param [in] surfaceFmt   type of surface (decoded image). Surface is output for decoder
	\param [in] maxWidth   maximum image width in pixels
	\param [in] maxHeight   maximum image height in pixels
	\param [in] checkBytestream   
	\param [out] dstBuffer   pointer for linked buffer for next component (output buffer of current component)
*/
extern fastStatus_t DLL fastJpegDecoderCreate(
    fastJpegDecoderHandle_t *handle,

    fastSurfaceFormat_t surfaceFmt,

    unsigned maxWidth,
    unsigned maxHeight,
	bool checkBytestream,

	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief Returns requested GPU memory for JPEG Decoder.

	Returns requested GPU memory for JPEG Decoder.
	Function returns requested memory size in Bytes for JPEG Decoder.
	
	\param [in] handle   JPEG Decoder handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastJpegDecoderGetAllocatedGpuMemorySize(
    fastJpegDecoderHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Decodes JPEG to surface.

	Decodes JPEG to surface.
	The procedure takes JPEG bytestream from h_Bytestream buffer in jfifInfo. Additional
	parameters for decoding are also taken from jfifInfo. Decoded surface is placed to output
	linked buffer and the following component of the pipeline consumes it. Struct
	fastJfifInfo_t for Decoder is populated by JfifLoad fuctions: fastJfifLoadFromFile
	and fastJfifLoadFromMemory.

	\param [in] handle   pointer to JPEG Decoder
	\param [in] jfifInfo   pointer to fastJfifInfo_t struct that contains all necessary information for JPEG decoding. For more detail see JPEG Encoder Description
*/
extern fastStatus_t DLL fastJpegDecode(
    fastJpegDecoderHandle_t handle,

	fastJfifInfo_t *jfifInfo
);

/*! \brief Destroys JPEG Decoder.

	Destroys JPEG Decoder.

	\param [in] handle   pointer to JPEG Decoder
*/
extern fastStatus_t DLL fastJpegDecoderDestroy(fastJpegDecoderHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Debayer functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates Debayer and returns associated handle.

	Creates Debayer and returns associated handle.
	Function fastCreateDebayerHandle allocates all necessary buffers in GPU memory.
	In case GPU does not have enough free memory, then fastCreateDebayerHandle will
	return FAST_INSUFFICIENT_DEVICE_MEMORY.
	Maximum dimensions of the image are set to Debayer during creation. Thus if
	transformation result exceeds the maximum value, then error status
	FAST_INVALID_SIZE will be returned.
	If component does not support current surface format then the function will
	return FAST_UNSUPPORTED_SURFACE.

	\param [out] handle   pointer to created Debayer handle
	\param [in] debayerType   debayer algorithm (HQLI, DFPD)
	\param [in] maxWidth   maximum image width in pixels
	\param [in] maxHeight   maximum image height in pixels
	\param [in] srcBuffer   linked buffer from previous component
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastDebayerCreate(
	fastDebayerHandle_t	*handle,

	fastDebayerType_t	debayerType,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief .

	.

	\param [in] handle   Debayer handle
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastDebayerChangeSrcBuffer(
	fastDebayerHandle_t	handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Returns requested GPU memory for Debayer.

	Returns requested GPU memory for Debayer.
	Function returns requested memory size in Bytes for Debayer
	
	\param [in] handle   Debayer handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastDebayerGetAllocatedGpuMemorySize(
    fastDebayerHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Restores image colors.

	Restores image colors.
	The procedure takes Bayer image from input linked buffer, restores colors
	based on pattern and algorithm and then stores color image to output linked buffer.
	If image size is greater than maximum value on creation, then error status
	FAST_INVALID_SIZE will be returned.

	\param [in] handle   Debayer handle 
	\param [in] bayerFmt   bayer pattern
	\param [in] width   image width in pixels
	\param [in] height   image height in pixels
*/
extern fastStatus_t DLL fastDebayerTransform(
 	fastDebayerHandle_t	handle,

	fastBayerPattern_t	bayerFmt,
	unsigned width,
    unsigned height
);

/*! \brief Destroys Debayer handle.

	Destroys Debayer handle.
	Procedure frees all device memory.

	\param [in] handle   Debayer handle
*/
extern fastStatus_t DLL fastDebayerDestroy(fastDebayerHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Resize functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates Resize and returns associated handle.

	Creates Resize and returns associated handle..
	Function fastResizeCreate allocates all necessary buffers in GPU memory.
	So in case GPU does not have enough free memory, then fastResizeCreate returns
	FAST_INSUFFICIENT_DEVICE_MEMORY.
	If component does not support current surface format then the function will return
	FAST_UNSUPPORTED_SURFACE.
	Parameter maxDstWidth has to be not more than maxSrcWidth, and maxDstHeight has to
	be not more than maxSrcHeight. In other case fastResizeCreate returns FAST_INVALID_SIZE.
	Also maxSrcWidth/maxDstWidth and maxSrcHeight/maxDstHeight have to be less or equal to
	maxScaleFactor. In other cases fastResizeCreate returns FAST_INVALID_SIZE.

	\param [out] handle   pointer to created Resizer component
	\param [in] maxSrcWidth   maximum input image width in pixels
	\param [in] maxSrcHeight   maximum input image height in pixels
	\param [in] maxDstWidth   maximum destination (cropped) image width in pixels
	\param [in] maxDstHeight   maximum destination (cropped) image height in pixels
	\param [in] maxScaleFactor   maximum scale factor (relation between source and destination dimensions)
	\param [in] shiftX   shift between source and destination grids by x coordinate. Currently ignored, should be 0,0
	\param [in] shiftY   shift between source and destination grids by y coordinate. Currently ignored, should be 0,0
	\param [in] srcBuffer   linked buffer from previous component
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastResizerCreate(
	fastResizerHandle_t *handle,

	unsigned maxSrcWidth,
	unsigned maxSrcHeight,

	unsigned maxDstWidth,
	unsigned maxDstHeight,

	double maxScaleFactor,

	float shiftX,
	float shiftY,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief .

	.

	\param [in] handle   Resizer handle
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastResizerChangeSrcBuffer(
	fastResizerHandle_t handle,

	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Resizes current image with preserving aspect ratio.

	Resizes current image with preserving aspect ratio.
	If size of input image or size of resized image are greater than maximum value on
	creation error status FAST_INVALID_SIZE will be returned.
	Height of resized image is calculated by the function and then customer application
	gets it in resizedHeight.	

	\param [in] handle   Resizer handle
	\param [in] resizeType   type of resize. Currently only FAST_LANCZOS resize is supported
	\param [in] width   input image width in pixels
	\param [in] height   input image height in pixels
	\param [in] resizedWidth   width of resized image in pixels
	\param [out] resizedHeight   height of resized image in pixels
*/
extern fastStatus_t DLL fastResizerTransform(
    fastResizerHandle_t	handle,
	fastResizeType_t resizeType,

 	unsigned width,
	unsigned height,

	unsigned resizedWidth,
	unsigned *resizedHeight
);

/*! \brief Resizes current image with preserving aspect ratio.

	Resizes current image with preserving aspect ratio.
	If size of input image or size of resized image are greater than maximum value on
	creation error status FAST_INVALID_SIZE will be returned.
	Height of resized image is calculated by the function and then customer application
	gets it in resizedHeight.	

	\param [in] handle   Resizer handle
	\param [in] resizeType   type of resize. Currently only FAST_LANCZOS resize is supported
	\param [in] width   input image width in pixels
	\param [in] height   input image height in pixels
	\param [in] background   
	\param [in] resizedWidth   width of resized image in pixels
	\param [out] resizedHeight   height of resized image in pixels
*/
extern fastStatus_t DLL fastResizerTransformWithPaddingCentered(
	fastResizerHandle_t	handle,
	fastResizeType_t resizeType,

	unsigned width,
	unsigned height,
	fastRgb_t background,

	unsigned resizedWidth,
	unsigned resizedHeight
);

/*! \brief Resizes current image without preserving aspect ratio.

	Resizes current image without preserving aspect ratio.
	If size of input image or size of resized image are greater than
	maximum value on creation error status FAST_INVALID_SIZE will be returned.
	Function allows upscale one dimension and downscale other dimension.

	\param [in] handle   Resizer handle
	\param [in] resizeType   type of resize. Currently only FAST_LANCZOS resize is supported
	\param [in] width   input image width in pixels
	\param [in] height   input image height in pixels
	\param [in] resizedWidth   width of resized image in pixels
	\param [out] resizedHeight   height of resized image in pixels
*/
extern fastStatus_t DLL fastResizerTransformStretch(
    fastResizerHandle_t	handle,
	fastResizeType_t resizeType,

 	unsigned width,
	unsigned height,

	unsigned resizedWidth,
	unsigned resizedHeight
);

/*! \brief Returns requested GPU memory for Resizer component.

	Calculate requested GPU memory for Resizer. 
	Function returns requested memory size in Bytes for Resizer component. 

	\param [in] handle   Resizer handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastResizerGetAllocatedGpuMemorySize(
    fastResizerHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Destroys Resizer component.

	Destroys Resizer component.
	Procedure frees all device memory.
	
	\param [in] handle   Resizer component handle
*/
extern fastStatus_t DLL fastResizerDestroy(fastResizerHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Image Filter functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates ImageFilter component and returns associated handle.

	Creates ImageFilter component and returns associated handle.
	Function fastImageFilterCreate allocates all necessary buffers in GPU memory. So in case
	GPU does not have enough free memory, then fastImageFilterCreate returns FAST_INSUFFICIENT_DEVICE_MEMORY.
	If component does not support current surface format then the function will return
	FAST_UNSUPPORTED_SURFACE.
	Pointer staticFilterParameters can point on the following structures: fastMad_t,
	fastBaseColorCorrection_t, fastWhiteBalance_t, fastToneCurve_t, fastColorSaturation_t and on
	all LUT structures.
	Filter FAST_GAUSSIAN_SHARPEN has no static parameters, so staticFilterParameters has to be null.
	
	\param [out] handle   pointer to created ImageFilter component
	\param [in] filterType   type of image filter
	\param [in] staticFilterParameters   static parameters for image filter
	\param [in] maxWidth   maximum image width in pixels
	\param [in] maxHeight   maximum image height in pixels
	\param [in] srcBuffer   linked buffer from previous component
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastImageFilterCreate(
    fastImageFiltersHandle_t  *handle,

	fastImageFilterType_t filterType,
	void *staticFilterParameters,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief .

	.

	\param [in] handle   ImageFilter component handle
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastImageFiltersChangeSrcBuffer(
	fastImageFiltersHandle_t handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Returns requested GPU memory for ImageFilter component.

	Returns requested GPU memory for ImageFilter component.
	Function returns requested memory size in Bytes for ImageFilter component.
	
	\param [in] handle   ImageFilter component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastImageFiltersGetAllocatedGpuMemorySize(
    fastImageFiltersHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Perform current ImageFilter transformation.

	Perform current ImageFilter transformation.
	If image size is greater than maximum value on creation error status
	FAST_INVALID_SIZE will be returned.
	Pointer filterParameters can point on the following structures:
	fastGaussianFilter_t, fastBaseColorCorrection_t, fastWhiteBalance_t, 
	fastToneCurve_t, fastColorSaturation_t and on all LUT structures.
	
	\param [in] handle   ImageFilter component handle
	\param [in] filterParameters   filter parameters for current image
	\param [in] width   image width in pixels
	\param [in] height   image height in pixels
*/
extern fastStatus_t DLL fastImageFiltersTransform(
 	fastImageFiltersHandle_t	handle,
	void *filterParameters,

 	unsigned width,
	unsigned height
);

/*! \brief Destroys ImageFilter component handle.

	Destroys ImageFilter component handle.
	Procedure frees all device memory.
	
	\param [in] handle   ImageFilter component handle
*/
extern fastStatus_t DLL fastImageFiltersDestroy(fastImageFiltersHandle_t handle);

////////////////////////////////////////////////////////////////////////////////
// Crop functions
////////////////////////////////////////////////////////////////////////////////
/*! \brief Creates Crop component and returns associated handle.

	Creates Crop component and returns associated handle.
	Function fastCropCreate allocates all necessary buffers in GPU memory. So in case GPU does
	not have enough free memory, then fastCropCreate will return FAST_INSUFFICIENT_DEVICE_MEMORY.
	Parameter maxDstWidth has to be not more than maxSrcWidth, and maxDstHeight has to be not more
	than maxSrcHeight. In other case fastCropCreate will return FAST_INVALID_SIZE.
	If component does not support current surface format then the function will return FAST_UNSUPPORTED_SURFACE.
	
	\param [out] handle   pointer to created Crop component
	\param [in] maxSrcWidth   maximum input image width in pixels
	\param [in] maxSrcHeight   maximum input image height in pixels
	\param [in] maxDstWidth   maximum destination (cropped) image width in pixels
	\param [in] maxDstHeight   maximum destination (cropped) image height in pixels
	\param [in] srcBuffer   linked buffer from previous component
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastCropCreate(
	fastCropHandle_t *handle,
	
	unsigned maxSrcWidth,
	unsigned maxSrcHeight,

	unsigned maxDstWidth,
	unsigned maxDstHeight,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief .

	.

	\param [in] handle   Crop component handle
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastCropChangeSrcBuffer(
	fastCropHandle_t handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Returns requested GPU memory for Crop component.

	Returns requested GPU memory for Crop component.
	Function returns requested memory size in Bytes for Crop component.
	
	\param [in] handle   Crop component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastCropGetAllocatedGpuMemorySize(
    fastCropHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Performs current Crop transformation.

	Performs current Crop transformation.
	If size of input image or size of cropped image greater than maximum value on
	creation error status FAST_INVALID_SIZE will be returned.
	Also leftTopCoordsX + croppedWidth has to be not more than width and
	leftTopCoordsY + croppedHeight has to be not more than height. In other case
	function returns FAST_INVALID_SIZE.
	
	\param [in] handle   Crop component handle
	\param [in] width   input image width in pixels
	\param [in] height   input image height in pixels
	\param [in] leftTopCoordsX   coordX in pixels for left top corner of cropped image in input image
	\param [in] leftTopCoordsY   coordY in pixels for left top corner of cropped image in input image
	\param [in] croppedWidth   cropped image width in pixels
	\param [in] croppedHeight   cropped image width in pixels
*/
extern fastStatus_t DLL fastCropTransform(
    fastCropHandle_t	handle,
	
  	unsigned width,
	unsigned height,

	unsigned leftTopCoordsX,
	unsigned leftTopCoordsY,
	unsigned croppedWidth,
	unsigned croppedHeight
);

/*! \brief Destroys Crop component handle.

	Destroys Crop component handle.
	Procedure frees all device memory.
	
	\param [in] handle   Crop component handle
*/
extern fastStatus_t DLL fastCropDestroy(fastCropHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Raw Unpacker calls (host)
///////////////////////////////////////////////////////////////////////////////
/*! \brief .

	.

	\param [out] handle   pointer to created RawImportFromHost component
	\param [in] rawFmt   _
	\param [in] surfaceFmt   _
	\param [in] maxWidth   maximum input image width in pixels
	\param [in] maxHeight   maximum input image height in pixels
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastRawImportFromHostCreate(
	fastRawUnpackerHandle_t	*handle,

	fastRawFormat_t	rawFmt,
	void* staticParameters,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief Returns requested GPU memory for RawImportFromHost component.

	Returns requested GPU memory for RawImportFromHost component.
	Function returns requested memory size in Bytes for RawImportFromHost component.

	\param [in] handle   RawImportFromHost handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastRawImportFromHostGetAllocatedGpuMemorySize(
	fastRawUnpackerHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief .

	.

	\param [in] handle   RawImportFromHost handle
	\param [in] src   _
	\param [in] srcPitch   _
	\param [in] width   _
	\param [in] height   _
*/
extern fastStatus_t DLL fastRawImportFromHostDecode(
	fastRawUnpackerHandle_t	handle,

	void* src,
	unsigned srcPitch,

	unsigned width,
	unsigned height
);

/*! \brief Destroys RawImportFromHost component handle.

	Destroys RawImportFromHost component handle.
	Procedure frees all device memory.

	\param [in] handle   RawImportFromHost component handle
*/
extern fastStatus_t DLL fastRawImportFromHostDestroy(fastRawUnpackerHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Raw Unpacker calls (device)
///////////////////////////////////////////////////////////////////////////////
/*! \brief .

	.

	\param [out] handle   pointer to created RawImportFromDevice component
	\param [in] rawFmt   _
	\param [in] surfaceFmt   _
	\param [in] maxWidth   maximum input image width in pixels
	\param [in] maxHeight   maximum input image height in pixels
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastRawImportFromDeviceCreate(
	fastRawUnpackerHandle_t	*handle,

	fastRawFormat_t	rawFmt,
	void* staticParameters,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief Returns requested GPU memory for RawImportFromDevice component.

	Returns requested GPU memory for RawImportFromDevice component.
	Function returns requested memory size in Bytes for RawImportFromDevice component.

	\param [in] handle   RawImportFromDevice handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastRawImportFromDeviceGetAllocatedGpuMemorySize(
	fastRawUnpackerHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief .

	.

	\param [in] handle   RawImportFromDevice handle
	\param [in] src   _
	\param [in] srcPitch   _
	\param [in] width   _
	\param [in] height   _
*/
extern fastStatus_t DLL fastRawImportFromDeviceDecode(
	fastRawUnpackerHandle_t	handle,

	void* src,
	unsigned srcPitch,

	unsigned width,
	unsigned height
);

/*! \brief Destroys RawImportFromDevice component handle.

	Destroys RawImportFromDevice component handle.
	Procedure frees all device memory.

	\param [in] handle   RawImportFromDevice component handle
*/
extern fastStatus_t DLL fastRawImportFromDeviceDestroy(fastRawUnpackerHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Bayer Merger functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates BayerMerger component and returns associated handle.

	Creates BayerMerger component and returns associated handle.
	Function fastBayerMergerCreate allocates all necessary buffers in GPU memory.
	So in case GPU does not have enough free memory, then fastBayerMergerCreate
	returns FAST_INSUFFICIENT_DEVICE_MEMORY.
	The function is different from other create functions because it takes maximum
	size of output image. BayerMerger takes splitted Bayer image and transforms it
	to normal Bayer image. It is more convenient for user to operate with size of
	restored (output) than splitted (input) image

	\param [out] handle   pointer to created BayerMerger component
	\param [in] maxDstWidth   maximum width of restored Bayer image in pixels
	\param [in] maxDstHeight   maximum height of restored Bayer image in pixels
	\param [in] srcBuffer   linked buffer from the previous component
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastBayerMergerCreate(
	fastBayerMergerHandle_t *handle,
	
	unsigned maxDstWidth,
	unsigned maxDstHeight,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief .

	.

	\param [in] handle   BayerMerger component handle
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastBayerMergerChangeSrcBuffer(
	fastBayerMergerHandle_t handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Restores Bayer image from splitted image.

	Restores Bayer image from splitted image.
	Function returns requested memory size in Bytes for BayerMerger component.

	\param [in] handle   BayerMerger component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastBayerMergerGetAllocatedGpuMemorySize(
    fastBayerMergerHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Returns requested GPU memory for BayerMerger component.

	Returns requested GPU memory for BayerMerger component.
	If image size is greater than maximum value on creation error status
	FAST_INVALID_SIZE will be returned.
	Restored Image width and height are taken from EXIF section, defined
	by SplitterExif_t structure in ExifInfo.hpp. Section is parsed by
	ParseSplitterExif function from ExifInfo.hpp.

	\param [in] handle   BayerMerger component handle
	\param [in] width   restored (original) image width in pixels
	\param [in] height   restored (original) image height in pixels
*/
extern fastStatus_t DLL fastBayerMergerMerge(
    fastBayerMergerHandle_t handle,
	
  	unsigned width,
	unsigned height
);

/*! \brief Destroys BayerMerger component.

	Destroys BayerMerger component.
	Procedure frees all device memory.

	\param [in] handle   BayerMerger component handle
*/
extern fastStatus_t DLL fastBayerMergerDestroy(fastBayerMergerHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Bayer Splitter functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates BayerSplitter component and returns associated handle.

	Creates BayerSplitter component and returns associated handle.
	Function fastBayerSplitterCreate allocates all necessary buffers in GPU memory.
	So in case GPU does not have enough free memory, fastBayerSplitterCreate returns
	FAST_INSUFFICIENT_DEVICE_MEMORY.
	If component does not support current surface format then the function will return
	FAST_UNSUPPORTED_SURFACE.
	Output parameters maxDstWidth and maxDstHeight are used by next pipeline component
	to determine its maxWidth and maxHeight. Intended next component is JPEG Encoder.

	\param [out] handle   pointer to created BayerSplitter component
	\param [in] maxSrcWidth   maximum input image width in pixels
	\param [in] maxSrcHeight   maximum input image height in pixels
	\param [out] maxDstWidth   maximum width of splitted Bayer image in pixels
	\param [out] maxDstHeight   maximum height of splitted Bayer image in pixels
	\param [in] srcBuffer   linked buffer from previous component
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastBayerSplitterCreate(
	fastBayerSplitterHandle_t *handle,
	
	unsigned maxSrcWidth,
	unsigned maxSrcHeight,

	unsigned *maxDstWidth,
	unsigned *maxDstHeight,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief .

	.

	\param [in] handle   BayerSplitter component handle
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastBayerSplitterChangeSrcBuffer(
	fastBayerSplitterHandle_t handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Returns requested GPU memory for BayerSplitter component.

	Returns requested GPU memory for BayerSplitter component.
	Function returns requested memory size in Bytes for BayerSplitter component.

	\param [in] handle   BayerSplitter component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastBayerSplitterGetAllocatedGpuMemorySize(
    fastBayerSplitterHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Function splits Bayer image on four planes.

	Function splits Bayer image on four planes.
	If image size is greater than maximum value on creation error status, FAST_INVALID_SIZE will be returned.
	Output parameters dstWidth and dstHeight are passed to the next component.

	\param [in] handle   BayerSplitter component handle
	\param [in] srcWidth   maximum input image width in pixels
	\param [in] srcHeight   maximum input image height in pixels
	\param [out] dstWidth   width of splitted Bayer image in pixels
	\param [out] dstHeight   height of splitted Bayer image in pixels
*/
extern fastStatus_t DLL fastBayerSplitterSplit(
    fastBayerSplitterHandle_t handle,

 	unsigned srcWidth,
	unsigned srcHeight,

	unsigned *dstWidth,
	unsigned *dstHeight
);

/*! \brief Destroys BayerSplitter component.

	Destroys BayerSplitter component.
	Procedure frees all device memory.

	\param [in] handle   BayerSplitter component handle
*/
extern fastStatus_t DLL fastBayerSplitterDestroy(fastBayerSplitterHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Pipeline import functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates ImportFromHostAdapter and returns associated handle.

	Creates ImportFromHostAdapter and returns associated handle.
	Function fastImportFromHostCreate allocates all necessary buffers in GPU memory.
	In case GPU does not have enough free memory, fastImportFromHostCreate returns
	FAST_INSUFFICIENT_DEVICE_MEMORY.

	\param [out] handle   pointer to created ImportFromHostAdapter handle.
	\param [in] surfaceFmt   defines input pixel format.
	\param [in] maxWidth   maximum image width in pixels
	\param [in] maxHeight   maximum image height in pixels.
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastImportFromHostCreate(
	fastImportFromHostHandle_t *handle,
	fastSurfaceFormat_t surfaceFmt,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief Copy image from CPU buffer to pipeline.

	Copy image from CPU buffer to pipeline.
	Buffer h_src has to be allocated with fastMalloc. Buffer allocated by original
	malloc also can be used, but copy speed will degrade.
	If image size is greater than maximum value on creation, then error status
	FAST_INVALID_SIZE will be returned.

	\param [in] handle   ImportFromHostAdapter handle.
	\param [in] h_src   pointer on CPU buffer with image data.
	\param [in] width   image width in pixels.
	\param [in] pitch   size of image row in Bytes.
	\param [in] height   image height in pixels.
*/
extern fastStatus_t DLL fastImportFromHostCopy(
	fastImportFromHostHandle_t handle,

	void* h_src,
	unsigned width,
	unsigned pitch,
	unsigned height
);

/*! \brief Returns requested GPU memory for ImportFromHostAdapter.

	Returns requested GPU memory for ImportFromHostAdapter.
	Function returns requested memory size in Bytes for ImportFromHostAdapter.

	\param [in] handle   ImportFromHostAdapter handle.
	\param [out] allocatedGpuSizeInBytes   memory size in Bytes.
*/
extern fastStatus_t DLL fastImportFromHostGetAllocatedGpuMemorySize(
    fastImportFromHostHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Destroy ImportFromHostAdapter.

	Destroy ImportFromHostAdapter.

	\param [in] handle   ImportFromHostAdapter handle.
*/
extern fastStatus_t DLL fastImportFromHostDestroy(fastImportFromHostHandle_t handle);

/*! \brief Creates ImportFromDeviceAdapter and returns associated handle.

	Creates ImportFromDeviceAdapter and returns associated handle.
	Function fastImportFromDeviceCreate allocates all necessary buffers in GPU memory.
	In case GPU does not have enough free memory, then fastImportFromDeviceCreate
	returns FAST_INSUFFICIENT_DEVICE_MEMORY.

	\param [out] handle   pointer to created ImportFromDeviceAdapter handle.
	\param [in] surfaceFmt   defines input pixel format.
	\param [in] maxWidth   maximum image width in pixels.
	\param [in] maxHeight   maximum image height in pixels.
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component).
*/
extern fastStatus_t DLL fastImportFromDeviceCreate(
	fastImportFromDeviceHandle_t *handle,
	fastSurfaceFormat_t surfaceFmt,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief Copy image from GPU buffer to pipeline.

	Copy image from GPU buffer to pipeline.
	Buffer d_src has to be allocated in Device memory by cudaMalloc.
	If image size is greater than maximum value on creation, then
	error status FAST_INVALID_SIZE will be returned.

	\param [in] handle   Import From Device Adapter handle.
	\param [in] d_src   pointer on Device buffer with image.
	\param [in] width   image width in pixels.
	\param [in] pitch   size of image row in Bytes.
	\param [in] height   image height in pixels.
*/
extern fastStatus_t DLL fastImportFromDeviceCopy(
	fastImportFromDeviceHandle_t handle,

	void* d_src,
	unsigned width,
	unsigned pitch,
	unsigned height
);

/*! \brief Returns requested GPU memory for ImportFromDeviceAdapter.

	Returns requested GPU memory for ImportFromDeviceAdapter.
	Function returns requested memory size in Bytes for ImportFromDeviceAdapter.

	\param [in] handle   ImportFromDeviceAdapter handle.
	\param [out] requestedGpuSizeInBytes   memory size in Bytes.
*/
extern fastStatus_t DLL fastImportFromDeviceGetAllocatedGpuMemorySize(
    fastImportFromDeviceHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Destroy ImportFromDeviceAdapter.

	Destroy ImportFromDeviceAdapter.

	\param [in] handle   ImportFromDeviceAdapter handle.
*/
extern fastStatus_t DLL fastImportFromDeviceDestroy(fastImportFromDeviceHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Pipeline export functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates ExportToHostAdapter and returns associated handle.

	Creates ExportToHostAdapter and returns associated handle.

	\param [out] handle   pointer to created ExportToHostAdapter handle.
	\param [out] surfaceFmt   pipeline output surface format.
	\param [in] srcBuffer   linked buffer from previous component.
*/
extern fastStatus_t DLL fastExportToHostCreate(
	fastExportToHostHandle_t *handle,
	fastSurfaceFormat_t *surfaceFmt,

	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Creates ExportToHostAdapter and returns associated handle.

	Creates ExportToHostAdapter and returns associated handle.

	\param [out] handle   pointer to created ExportToHostAdapter handle.
	\param [out] surfaceFmt   pipeline output surface format.
	\param [in] srcBuffer   linked buffer from previous component.
*/
extern fastStatus_t DLL fastExportToHostExclusiveCreate(
	fastExportToHostHandle_t* handle,
	fastSurfaceFormat_t* surfaceFmt,

	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief .

	.

	\param [in] handle   ExportToHostAdapter component handle
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastExportToHostChangeSrcBuffer(
	fastExportToHostHandle_t handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Copies image from pipeline to CPU buffer.

	Copies image from pipeline to CPU buffer.
	Buffer h_dst has to be allocated with fastMalloc. Buffer, which is allocated
	by original malloc also can be used, but copy speed will degrade. If size
	of h_dst is not enough, then the function will fail with segmentation fault.
	Export Parameters allows to convert RGB color format to BGR color format. In
	other case parameters have to be null. To convert color format to BGR convert
	member of fastExportParameters_t have to set in FAST_CONVERT_BGR.

	\param [in] handle   ExportToHostAdapter handle.
	\param [in] h_dst   pointer on Host buffer for image.
	\param [in] width   image width in pixels.
	\param [in] pitch   size of image row in Bytes.
	\param [in] height   image height in pixels.
	\param [in] parameters   export parameters.
*/
extern fastStatus_t DLL fastExportToHostCopy(
	fastExportToHostHandle_t handle,
	void* h_dst,

	unsigned width,
	unsigned pitch,
	unsigned height,

	fastExportParameters_t *parameters
);

/*! \brief Returns requested GPU memory for ExportToHostAdapter.

	Returns requested GPU memory for ExportToHostAdapter.
	Function returns requested memory size in Bytes for ExportToHostAdapter.

	\param [in] handle   ExportToHostAdapter handle.
	\param [out] requestedGpuSizeInBytes   memory size in Bytes.
*/
extern fastStatus_t DLL fastExportToHostGetAllocatedGpuMemorySize(
	fastExportToHostHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Destroy ExportToHostAdapter.

	Destroy ExportToHostAdapter.

	\param [in] handle   ExportToHostAdapter handle.
*/
extern fastStatus_t DLL fastExportToHostDestroy(fastExportToHostHandle_t handle);

/*! \brief Creates ExportToDeviceAdapter and returns associated handle.

	Creates ExportToDeviceAdapter and returns associated handle.

	\param [out] handle   pointer to created ExportToDeviceAdapter handle.
	\param [out] surfaceFmt   pipeline output surface format.
	\param [in] srcBuffer   linked buffer from previous component.
*/
extern fastStatus_t DLL fastExportToDeviceCreate(
	fastExportToDeviceHandle_t *handle,
	fastSurfaceFormat_t *surfaceFmt,

	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Creates ExportToDeviceAdapter and returns associated handle.

	Creates ExportToDeviceAdapter and returns associated handle.

	\param [out] handle   pointer to created ExportToDeviceAdapter handle.
	\param [out] surfaceFmt   pipeline output surface format.
	\param [in] srcBuffer   linked buffer from previous component.
*/
extern fastStatus_t DLL fastExportToDeviceExclusiveCreate(
	fastExportToDeviceHandle_t* handle,
	fastSurfaceFormat_t* surfaceFmt,

	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief .

	.

	\param [in] handle   fastExportToDeviceHandle_t handle.
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastExportToDeviceChangeSrcBuffer(
	fastExportToDeviceHandle_t handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Copy image from pipeline to GPU buffer.

	Copy image from pipeline to GPU buffer.
	Buffer d_dst has to be allocated in Device memory by cudaMalloc. If size of d_dst is
	not enough, then function will fail with segmentation fault.
	Export Parameters allows to convert RGB color format to BGR color format. In other case
	parameters have to be null. To convert color format to BGR convert member of
	fastExportParameters_t have to set in FAST_CONVERT_BGR.

	\param [in] handle   ExportToDeviceAdapter handle.
	\param [in] d_dst   pointer on Device buffer for image.
	\param [in] width   image width in pixels.
	\param [in] pitch   size of image row in Bytes.
	\param [in] height   image height in pixels.
	\param [in] parameters   export parameters.
*/
extern fastStatus_t DLL fastExportToDeviceCopy(
	fastExportToDeviceHandle_t handle,
	void* d_dst,

	unsigned width,
	unsigned pitch,
	unsigned height,

	fastExportParameters_t *parameters
);

/*! \brief .

	.

	\param [in] handle   fastExportToDeviceHandle_t handle.
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastExportToDeviceGetAllocatedGpuMemorySize(
	fastExportToDeviceHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Destroy ExportToDeviceAdapter.

	Destroy ExportToDeviceAdapter.

	\param [in] handle   ExportToDeviceAdapter handle.
*/
extern fastStatus_t DLL fastExportToDeviceDestroy(fastExportToDeviceHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Affine functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates Affine transformation component and returns associated handle.

	Creates Affine transformation component and returns associated handle.
	Function fastAffineCreate allocates all necessary buffers in GPU memory. So in
	case GPU does not have enough free memory, then fastAffineCreate will return
	FAST_INSUFFICIENT_DEVICE_MEMORY.
	There are 5 currently supported affine transformations: Flip, Flop, Rotate 180,
	Rotate 90 to left, Rotate 90 to right. All affine transformations are applicable
	for gray and for color images. Rotation 90 left and Rotation 90 right change image
	dimensions: width becomes height, height becomes width. So maxWidth and maxHeight
	of the following component have to be properly adjusted.
	If component does not support current surface format then the function will return
	FAST_UNSUPPORTED_SURFACE.

	\param [out] handle   pointer to created Affine component
	\param [in] affineType   type of affine transformation
	\param [in] affineTransforms   possible affine transformations
	\param [in] maxWidth   maximum input image width in pixels
	\param [in] maxHeight   maximum input image height in pixels
	\param [in] srcBuffer   linked buffer from previous component
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastAffineCreate(
    fastAffineHandle_t  *handle,

	unsigned char affineTransformMask,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief .

	.

	\param [in] handle   Affine component handle.
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastAffineChangeSrcBuffer(
	fastAffineHandle_t	handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Returns requested GPU memory for Affine transformation component.

	Returns requested GPU memory for Affine transformation component.
	Function returns requested memory size in Bytes for Affine component.

	\param [in] handle   Affine component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastAffineGetAllocatedGpuMemorySize(
    fastAffineHandle_t	handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Performs current Affine transformation.

	Performs current Affine transformation.
	If image size is greater than maximum value on creation, then error
	status FAST_INVALID_SIZE will be returned.

	\param [in] handle   Affine component handle
	\param [in] affineType   type of affine transformation
	\param [in] width   image width in pixels
	\param [in] height   image height in pixels
*/
extern fastStatus_t DLL fastAffineTransform(
 	fastAffineHandle_t	handle,

	fastAffineTransformations_t affineType,
 	unsigned width,
	unsigned height
);

/*! \brief Destroys Affine component handle.

	Destroys Affine component handle.
	Procedure frees all device memory.

	\param [in] handle   Affine component handle
*/
extern fastStatus_t DLL fastAffineDestroy(fastAffineHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Mux functions
///////////////////////////////////////////////////////////////////////////////
/*! \brief Creates Mux and returns associated handle.

	Creates Mux and returns associated handle.
	Allocates necessary buffers in GPU memory. In case GPU does not have
	enough free memory returns FAST_INSUFFICIENT_DEVICE_MEMORY.
	All input buffers have to be same size and type, else function returns
	FAST_INVALID_FORMAT.

	\param [out] handle   pointer to created Mux handle
	\param [in] srcBuffers   array of linked buffer from previous component
	\param [in] numberOfInputs   element count in srcBuffers
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastMuxCreate(
	fastMuxHandle_t *handle,

	fastDeviceSurfaceBufferHandle_t* srcBuffers,
	unsigned numberOfInputs,

	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief Selects specified input and passes it to the output.

	Selects specified input and passes it to the output.
	Index is zero-based numbering. It has to be less than
	numberOfInputs in Create function, else function returns
	FAST_INVALID_SIZE.

	\param [in] handle   Mux handle
	\param [in] srcBufferIndex   index of selected input
*/
extern fastStatus_t DLL fastMuxSelect(
	fastMuxHandle_t handle,

	unsigned srcBufferIndex
);

/*! \brief Destroys Mux.

	Destroys Mux.
	Procedure frees all device memory.

	\param [in] handle   Mux handle
*/
extern fastStatus_t DLL fastMuxDestroy(fastMuxHandle_t handle);

////////////////////////////////////////////////////////////////////////////////
// SDI Import From Host
////////////////////////////////////////////////////////////////////////////////
/*! \brief Creates SDI Import component and returns associated handle.

	Creates SDI Import component and returns associated handle.
	Allocates necessary buffers in GPU memory. In case GPU does not
	have enough free memory returns FAST_INSUFFICIENT_DEVICE_MEMORY.
	If component does not support current surface format, then the
	function will return FAST_UNSUPPORTED_SURFACE.

	\param [out] handle   pointer to created SDI Import handle
	\param [in] sdiFmt   SDI format
	\param [in] maxWidth   maximum width of image in pixels
	\param [in] maxHeight   maximum height of image in pixels
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastSDIImportFromHostCreate(
	fastSDIImportFromHostHandle_t *handle,

	fastSDIFormat_t	sdiFmt,
	void* staticParameters,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief Returns requested GPU memory size for SDI Import component.

	Function returns requested memory size in Bytes for SDI Import component.

	\param [in] handle   SDI Import component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastSDIImportFromHostGetAllocatedGpuMemorySize(
	fastSDIImportFromHostHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Loads SDI formatted image to the pipeline.

	Loads SDI formatted image to the pipeline.
	Buffer h_src has to be allocated by fastMalloc. Buffer allocated by
	original malloc also can be used, but copy speed will degrade.
	If image size is greater than maximum value on creation, then error
	status FAST_INVALID_SIZE will be returned.

	\param [in] handle   SDI Import component handle
	\param [in] h_src   SDI formatted image
	\param [in] width   image width in pixels
	\param [in] height   image height in pixels
*/
extern fastStatus_t DLL fastSDIImportFromHostCopy(
	fastSDIImportFromHostHandle_t handle,
	
	void* h_src,
	unsigned width,
    unsigned height
);

/*! \brief Loads SDI formatted image to the pipeline.

	Loads SDI formatted image to the pipeline.
	Buffer h_src has to be allocated by fastMalloc. Buffer allocated by
	original malloc also can be used, but copy speed will degrade.
	If image size is greater than maximum value on creation, then error
	status FAST_INVALID_SIZE will be returned.

	\param [in] handle   SDI Import component handle
	\param [in] h_src   SDI formatted image
    \param [in] pitch   
	\param [in] width   image width in pixels
	\param [in] height   image height in pixels
*/
extern fastStatus_t DLL fastSDIImportFromHostCopyPacked(
	fastSDIImportFromHostHandle_t handle,
	
	void* h_src,
	unsigned pitch,

	unsigned width,
    unsigned height
);

/*! \brief .

	.

	\param [in] handle   SDI Import component handle.
	\param [in] srcY   _
	\param [in] srcU   _
	\param [in] srcV   _
*/
extern fastStatus_t DLL fastSDIImportFromHostCopy3(
	fastSDIImportFromHostHandle_t handle,

	fastChannelDescription_t *srcY,
	fastChannelDescription_t *srcU,
	fastChannelDescription_t *srcV
);

/*! \brief Destroys SDI Import component.

	Destroys SDI Import component.
	Procedure frees all device memory.

	\param [in] handle   SDI Import component handle
*/
extern fastStatus_t DLL fastSDIImportFromHostDestroy(fastSDIImportFromHostHandle_t handle);

////////////////////////////////////////////////////////////////////////////////
// SDI Import From Device
////////////////////////////////////////////////////////////////////////////////
/*! \brief .

	.

	\param [out] handle   pointer to created SDI Import handle
	\param [in] sdiFmt   SDI format
	\param [in] maxWidth   maximum width of image in pixels
	\param [in] maxHeight   maximum height of image in pixels
	\param [out] dstBuffer   pointer for linked buffer for the next component (output buffer of current component)
*/
extern fastStatus_t DLL fastSDIImportFromDeviceCreate(
	fastSDIImportFromDeviceHandle_t *handle,

	fastSDIFormat_t	sdiFmt,
	void* staticParameters,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief Returns requested GPU memory size for SDI Import component.

	Function returns requested memory size in Bytes for SDI Import component.

	\param [in] handle   SDI Import component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastSDIImportFromDeviceGetAllocatedGpuMemorySize(
	fastSDIImportFromDeviceHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief .

	.

	\param [in] handle   SDI Import component handle
	\param [in] h_src   SDI formatted image
	\param [in] width   image width in pixels
	\param [in] height   image height in pixels
*/
extern fastStatus_t DLL fastSDIImportFromDeviceCopy(
	fastSDIImportFromDeviceHandle_t handle,
	
	void* h_src,
	unsigned width,
    unsigned height
);

/*! \brief .

	.

	\param [in] handle   SDI Import component handle
	\param [in] h_src   SDI formatted image
	\param [in] pitch   
	\param [in] width   image width in pixels
	\param [in] height   image height in pixels
*/
extern fastStatus_t DLL fastSDIImportFromDeviceCopyPacked(
	fastSDIImportFromDeviceHandle_t handle,

	void* h_src,
	unsigned pitch,

	unsigned width,
	unsigned height
);

/*! \brief .

	.

	\param [in] handle   SDI Import component handle.
	\param [in] srcY   _
	\param [in] srcU   _
	\param [in] srcV   _
*/
extern fastStatus_t DLL fastSDIImportFromDeviceCopy3(
	fastSDIImportFromDeviceHandle_t handle,

	fastChannelDescription_t *srcY,
	fastChannelDescription_t *srcU,
	fastChannelDescription_t *srcV
);

/*! \brief Destroys SDI Import component.

	Destroys SDI Import component.
	Procedure frees all device memory.

	\param [in] handle   SDI Import component handle
*/
extern fastStatus_t DLL fastSDIImportFromDeviceDestroy(fastSDIImportFromDeviceHandle_t handle);

////////////////////////////////////////////////////////////////////////////////
// SDI Export To Host
////////////////////////////////////////////////////////////////////////////////
/*! \brief Creates SDI Export component and returns associated handle.

	Creates SDI Export component and returns associated handle.
	Allocates necessary buffers in GPU memory. In case GPU does not have
	enough free memory returns FAST_INSUFFICIENT_DEVICE_MEMORY.
	If srcBuffer format is incompatible with selected sdiFmt, then
	function returns FAST_INVALID_FORMAT.

	\param [out] handle   pointer to created SDI Import handle
	\param [in] sdiFmt   SDI format
	\param [out] surfaceFmt   
	\param [in] maxWidth   maximum width of image in pixels
	\param [in] maxHeight   maximum height of image in pixels
	\param [out] srcBuffer   linked buffer from previous component
*/
extern fastStatus_t DLL fastSDIExportToHostCreate(
	fastSDIExportToHostHandle_t *handle,

	fastSDIFormat_t	sdiFmt,
	void *staticParameters,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief .

	.

	\param [in] handle   SDI Export component handle.
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastSDIExportToHostChangeSrcBuffer(
	fastSDIExportToHostHandle_t handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Returns requested GPU memory size for SDI Export component.

	Function returns requested memory size in Bytes for SDI Export component.

	\param [in] handle   SDI Export component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastSDIExportToHostGetAllocatedGpuMemorySize(
	fastSDIExportToHostHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief Exports SDI formatted image from pipeline to host memory.

	Exports SDI formatted image from pipeline to host memory.
	Buffer size in Bytes can be calculated by GetSDIBufferSize function
	from HelperSDI.hpp (part of SDIConverterSample application).
	Buffer h_dst has to be allocated by fastMalloc. Buffer allocated by
	original malloc also can be used, but copy speed will degrade.
	User has to estimate width and height of export image and allocate
	buffer according to these values. Function returns real width and height.

	\param [in] handle   SDI Export component handle
	\param [out] h_dst   buffer for exported SDI formatted image
	\param [out] width   image width in pixels
	\param [out] height   image height in pixels
*/
extern fastStatus_t DLL fastSDIExportToHostCopy(
	fastSDIExportToHostHandle_t handle,

	void* h_dst,
	unsigned *width,
	unsigned *height
);

/*! \brief .

	.

	\param [in] handle   SDI Export component handle.
	\param [in] dstY   _
	\param [in] dstU   _
	\param [in] dstV   _
*/
extern fastStatus_t DLL fastSDIExportToHostCopy3(
	fastSDIExportToHostHandle_t handle,

	fastChannelDescription_t *dstY,
	fastChannelDescription_t *dstU,
	fastChannelDescription_t *dstV
);

/*! \brief Destroys SDI Export component.

	Destroys SDI Export component.
	Procedure frees all device memory.

	\param [in] handle   SDI Export component handle
*/
extern fastStatus_t DLL fastSDIExportToHostDestroy(fastSDIExportToHostHandle_t handle);

////////////////////////////////////////////////////////////////////////////////
// SDI Export To Device
////////////////////////////////////////////////////////////////////////////////
/*! \brief .

	.

	\param [out] handle   pointer to created SDI Export handle
	\param [in] sdiFmt   SDI format
	\param [in] staticParameters
	\param [in] maxWidth   maximum width of image in pixels
	\param [in] maxHeight   maximum height of image in pixels
	\param [out] srcBuffer   linked buffer from previous component
*/
extern fastStatus_t DLL fastSDIExportToDeviceCreate(
	fastSDIExportToDeviceHandle_t *handle,

	fastSDIFormat_t	sdiFmt,
	void *staticParameters,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief .

	.

	\param [in] handle   SDI Export component handle.
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastSDIExportToDeviceChangeSrcBuffer(
	fastSDIExportToDeviceHandle_t handle,

	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Returns requested GPU memory size for SDI Export component.

	Function returns requested memory size in Bytes for SDI Export component.

	\param [in] handle   SDI Export component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastSDIExportToDeviceGetAllocatedGpuMemorySize(
	fastSDIExportToDeviceHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief .

	.

	\param [in] handle   SDI Export component handle
	\param [out] h_dst   buffer for exported SDI formatted image
	\param [out] width   image width in pixels
	\param [out] height   image height in pixels
*/
extern fastStatus_t DLL fastSDIExportToDeviceCopy(
	fastSDIExportToDeviceHandle_t handle,

	void* h_dst,
	unsigned *width,
	unsigned *height
);

/*! \brief .

	.

	\param [in] handle   SDI Export component handle.
	\param [in] dstY   _
	\param [in] dstU   _
	\param [in] dstV   _
*/
extern fastStatus_t DLL fastSDIExportToDeviceCopy3(
	fastSDIExportToDeviceHandle_t handle,

	fastChannelDescription_t *dstY,
	fastChannelDescription_t *dstU,
	fastChannelDescription_t *dstV
);

/*! \brief Destroys SDI Export component.

	Destroys SDI Export component.
	Procedure frees all device memory.

	\param [in] handle   SDI Export component handle
*/
extern fastStatus_t DLL fastSDIExportToDeviceDestroy(fastSDIExportToDeviceHandle_t handle);

///////////////////////////////////////////////////////////////////////////////
// Surface Converter calls
///////////////////////////////////////////////////////////////////////////////
/*! \brief .

	.

	\param [out] handle   pointer to created Surface Converter handle
	\param [in] surfaceConverterType   _
	\param [in] staticSurfaceConverterParameters   _
	\param [in] maxWidth   maximum width of image in pixels
	\param [in] maxHeight   maximum height of image in pixels
	\param [in] srcBuffer   linked buffer from previous component
	\param [out] dstBuffer   _
*/
extern fastStatus_t DLL fastSurfaceConverterCreate(
	fastSurfaceConverterHandle_t *handle,

	fastSurfaceConverter_t surfaceConverterType,
	void *staticSurfaceConverterParameters,

	unsigned maxWidth,
	unsigned maxHeight,

	fastDeviceSurfaceBufferHandle_t  srcBuffer,
	fastDeviceSurfaceBufferHandle_t *dstBuffer
);

/*! \brief .

	.

	\param [in] handle   Surface Converter component handle.
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastSurfaceConverterChangeSrcBuffer(
	fastSurfaceConverterHandle_t handle,
	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief Returns requested GPU memory size for Surface Converter component.

	Function returns requested memory size in Bytes for Surface Converter component.

	\param [in] handle   Surface Converter component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastSurfaceConverterGetAllocatedGpuMemorySize(
	fastSurfaceConverterHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief .

	.

	\param [in] handle   Surface Converter component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastSurfaceConverterTransform(
	fastSurfaceConverterHandle_t handle,
	void *surfaceConverterParameters,

	unsigned width,
	unsigned height
);

/*! \brief Destroys Surface Converter component.

	Destroys Surface Converter component.
	Procedure frees all device memory.

	\param [in] handle   Surface Converter component handle
*/
extern fastStatus_t DLL fastSurfaceConverterDestroy(fastSurfaceConverterHandle_t handle);

////////////////////////////////////////////////////////////////////////////////
// Histogram calls
////////////////////////////////////////////////////////////////////////////////
/*! \brief .

	.

	\param [out] handle   pointer to created Histogram handle
	\param [in] histogramType   _
	\param [in] staticParameters   _
	\param [in] bins   _
	\param [in] maxWidth   maximum width of image in pixels
	\param [in] maxHeight   maximum height of image in pixels
	\param [in] srcBuffer   linked buffer from previous component
*/
extern fastStatus_t DLL fastHistogramCreate(
	fastHistogramHandle_t *handle,

	fastHistogramType_t histogramType,
	void *staticParameters,
	unsigned int bins,

	unsigned int maxWidth,
	unsigned int maxHeight,

	fastDeviceSurfaceBufferHandle_t  srcBuffer
);

/*! \brief Returns requested GPU memory size for Histogram component.

	Function returns requested memory size in Bytes for Histogram component.

	\param [in] handle   Histogram component handle
	\param [out] requestedGpuSizeInBytes   memory size in Bytes
*/
extern fastStatus_t DLL fastHistogramGetAllocatedGpuMemorySize(
	fastHistogramHandle_t handle,

	size_t *requestedGpuSizeInBytes
);

/*! \brief .

	.

	\param [in] handle   Histogram component handle.
	\param [in] srcBuffer   _
*/
extern fastStatus_t DLL fastHistogramChangeSrcBuffer(
	fastHistogramHandle_t handle,

	fastDeviceSurfaceBufferHandle_t srcBuffer
);

/*! \brief .

	.

	\param [in] handle   Histogram component handle
	\param [in] histogramParameters   _
	\param [in] roiLeftTopX   _
	\param [in] roiLeftTopY   _
	\param [in] roiWidth   _
	\param [in] roiHeight   _
	\param [out] h_histogram   _
*/
extern fastStatus_t DLL fastHistogramCalculate(
	fastHistogramHandle_t handle,
	void *histogramParameters,
	unsigned int roiLeftTopX,
	unsigned int roiLeftTopY,
	unsigned int roiWidth,
	unsigned int roiHeight,
	unsigned int *h_histogram
);

/*! \brief Destroys Histogram component.

	Destroys Histogram component.
	Procedure frees all device memory.

	\param [in] handle   Histogram component handle
*/
extern fastStatus_t DLL fastHistogramDestroy(fastHistogramHandle_t handle);

////////////////////////////////////////////////////////////////////////////////
// License Info
////////////////////////////////////////////////////////////////////////////////

extern fastStatus_t DLL fastLicenseInfo(
    fastLicenseInfo_t *licenseInfo
);


extern fastStatus_t DLL fastDeviceSurfaceBufferStubCreate(
	fastSurfaceFormat_t surfaceFormat,
	unsigned maxWidth,
	unsigned maxHeight,
	fastDeviceSurfaceBufferHandle_t* dstBuffer 
);

extern fastStatus_t DLL fastDeviceSurfaceBufferStubDestroy(
	fastDeviceSurfaceBufferHandle_t* bufferStub
);


#ifdef __cplusplus
}
#endif

#endif
