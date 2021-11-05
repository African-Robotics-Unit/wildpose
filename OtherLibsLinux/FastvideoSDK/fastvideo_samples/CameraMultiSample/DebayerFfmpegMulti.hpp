/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __DEBAYER_FFMPEG_MULTI__
#define __DEBAYER_FFMPEG_MULTI__

#include "fastvideo_sdk.h"
#include "fastvideo_mjpeg.h"

#include "Image.h"
#include "FastAllocator.h"
#include "CameraMultiSampleOptions.h"

class DebayerFfmpegMulti
{
private:
	static const int MaxWritersCount = 2;
	static const int WorkItemQueueLength = 20;

	CameraMultiSampleOptions options_;

	fastImportFromHostHandle_t hImportFromHost;

	fastImageFiltersHandle_t hSam_0;
	fastImageFiltersHandle_t hSam_1;
	fastMuxHandle_t hSamMux;

	fastImageFiltersHandle_t hColorCorrection;
	fastDebayerHandle_t hDebayer;

	fastImageFiltersHandle_t hLut_0;
	fastImageFiltersHandle_t hLut_1;
	fastMuxHandle_t hLutMux;

	fastJpegEncoderHandle_t hEncoder;
	fastMJpegAsyncWriterHandle_t hMjpeg;
	
	fastDeviceSurfaceBufferHandle_t srcBuffer;
	/*
	 * madMuxBuffer[0] - import from host buffer
	 * madMuxBuffer[1] - hSam_0 buffer
	 * madMuxBuffer[2] - hSam_1 buffer
	 */
	fastDeviceSurfaceBufferHandle_t madMuxBuffer[3];
	fastDeviceSurfaceBufferHandle_t colorCorrectionBuffer;
	fastDeviceSurfaceBufferHandle_t debayerBuffer;
	fastDeviceSurfaceBufferHandle_t lutBuffer;
	/*
	 * lutMuxBuffer[0] - hLut_0 buffer
	 * lutMuxBuffer[1] - hLut_1 buffer
	 */
	fastDeviceSurfaceBufferHandle_t lutMuxBuffer[2];

	int FileIndex0;
	int FileIndex1;

	bool info;

public:
	DebayerFfmpegMulti(bool info) {
		this->info = info; 
		hMjpeg = NULL;
		hSam_0 = hSam_1 = NULL;
	};
	~DebayerFfmpegMulti(void) {};

	fastStatus_t Init(
		CameraMultiSampleOptions &options,
		std::unique_ptr<unsigned char, FastAllocator> &lut_0, float *matrixA_0, char *matrixB_0,
		std::unique_ptr<unsigned char, FastAllocator> &lut_1, float *matrixA_1, char *matrixB_1
	);
	fastStatus_t StoreFrame(Image<FastAllocator> &image, int cameraId);
	fastStatus_t Close(void) const;
};

#endif	// __DEBAYER_FFMPEG_MULTI__
