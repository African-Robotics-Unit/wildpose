/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#ifndef __HISTOGRAM_H__
#define __HISTOGRAM_H__

#include <list>
#include <memory>

#include "Image.h"
#include "FastAllocator.h"

#include "fastvideo_sdk.h"
#include "HistogramSampleOptions.h"
#include <cuda_runtime.h>

class Histogram {
private:
	fastImportFromHostHandle_t hImportFromHost;
	fastHistogramHandle_t hHistogram;
	fastDeviceSurfaceBufferHandle_t srcBuffer;

	std::unique_ptr<unsigned int, FastAllocator> histCPU;

	fastBayerPatternParam_t histogramBayer;
	fastHistogramParade_t histogramParad;

	cudaEvent_t histogramEnabled;

	HistogramSampleOptions options;
	fastSurfaceFormat_t surfaceFmt;
	bool info;

	cudaEvent_t event;
	void SaveHistogramToFile(const char *fname) const;
	void SaveParadeToFile(const char *fname) const;

public:
	Histogram(void) { };
	~Histogram(void) { };

	fastStatus_t Init(HistogramSampleOptions &options);
	fastStatus_t Calculate(std::list< Image<FastAllocator> > &image) const;
	fastStatus_t Close() const;
}; /* Histogram */
		

#endif /* __HISTOGRAM_H__ */
