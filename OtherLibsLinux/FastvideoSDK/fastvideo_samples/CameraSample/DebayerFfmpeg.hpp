/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __DEBAYER_FFMPEG_H__
#define __DEBAYER_FFMPEG_H__

#include "fastvideo_sdk.h"
#include "fastvideo_mjpeg.h"

#include "Image.h"
#include "FastAllocator.h"
#include "CameraSampleOptions.h"

class DebayerFfmpeg
{
private:
	static const int MaxWritersCount = 1;
	static const int WorkItemQueueLength = 20;

	fastImportFromHostHandle_t hImportFromHost;
	fastImageFiltersHandle_t hSam;
	fastImageFiltersHandle_t hColorCorrection;
	fastDebayerHandle_t hDebayer;
	fastImageFiltersHandle_t hLut;
	fastJpegEncoderHandle_t hEncoder;
	fastExportToDeviceHandle_t hExportToDevice;
	fastMJpegAsyncWriterHandle_t hMjpeg;
	
	fastDeviceSurfaceBufferHandle_t srcBuffer;
	fastDeviceSurfaceBufferHandle_t madBuffer;
	fastDeviceSurfaceBufferHandle_t colorCorrectionBuffer;
	fastDeviceSurfaceBufferHandle_t debayerBuffer;
	fastDeviceSurfaceBufferHandle_t lutBuffer;

	unsigned maxWidth;
	unsigned maxHeight;

	int FileIndex;
	int frameRepeat;
	int frameRate;

	unsigned char *d_buffer;

	bool info;

	int quality;
	unsigned restartInterval;
	fastJpegFormat_t jpegFmt;
	fastSurfaceFormat_t surfaceFmt;
	fastBayerPattern_t bayer_pattern_;
public:
	DebayerFfmpeg(bool info) { this->info = info; hMjpeg = NULL; hSam = NULL; };
	~DebayerFfmpeg(void) { };

	fastStatus_t Init(CameraSampleOptions &options, std::unique_ptr<unsigned char, FastAllocator> &lut, float *matrixA, char *matrixB);
	fastStatus_t StoreFrame(Image<FastAllocator> &image);
	fastStatus_t Close() const;

	void *GetDevicePtr();
};

#endif	// __DEBAYER_FFMPEG_H__
