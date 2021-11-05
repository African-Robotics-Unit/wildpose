/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __CAMERA_MULTI_SAMPLE_OPTIONS__
#define __CAMERA_MULTI_SAMPLE_OPTIONS__

#include "fastvideo_sdk.h"

#include "BaseOptions.h"
#include "JpegEncoderOptions.h"
#include "DebayerOptions.h"
#include "BaseColorCorrectionOptions.h"
#include "GrayscaleCorrectionOptions.h"
#include "FfmpegOptions.h"

class CameraMultiSampleOptions : public virtual BaseOptions {
public:
	char *OutputPath_2;

	JpegEncoderOptions JpegEncoder;
	DebayerOptions Debayer;
	BaseColorCorrectionOptions BaseColorCorrection_0;
	GrayscaleCorrectionOptions MAD_0;

	BaseColorCorrectionOptions BaseColorCorrection_1;
	GrayscaleCorrectionOptions MAD_1;
	char *Lut_1;

	FfmpegOptions FFMPEG;

	virtual bool Parse(int argc, char *argv[]);

	CameraMultiSampleOptions(void) : BaseColorCorrection_1(1), MAD_1(1) {};
	~CameraMultiSampleOptions(void) {};
};

#endif // __CAMERA_MULTI_SAMPLE_OPTIONS__