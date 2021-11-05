/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __JPEG_ENCODER_OPTIONS__
#define __JPEG_ENCODER_OPTIONS__

#include "fastvideo_sdk.h"

class JpegEncoderOptions {
private:
	fastJpegFormat_t ParseSubsamplingFmt(const int samplingFmtCode);

public:
	unsigned Quality;
	unsigned RestartInterval;
	fastJpegFormat_t SamplingFmt;

	bool GrayAsRGB;
	bool BayerCompression;

	bool Async;
	char *QuantTableFileName;
	bool noExif;

	JpegEncoderOptions(void) {
		BayerCompression = false;
	};
	~JpegEncoderOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __JPEG_ENCODER_OPTIONS__