/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "Histogram.h"

#include <cuda_runtime_api.h>

#include "HistogramTraits.hpp"
#include "checks.h"

fastStatus_t Histogram::Init(HistogramSampleOptions &options) {
	/* Populate values for class members */
	surfaceFmt = options.SurfaceFmt;
	info = options.Info;
	this->options = options;

	CHECK_FAST(fastImportFromHostCreate(
		&hImportFromHost,
		options.SurfaceFmt,
		options.MaxWidth,
		options.MaxHeight,
		&srcBuffer
	));

	if (options.SurfaceFmt == FAST_BGR8) {
		options.SurfaceFmt = FAST_RGB8;
	}

	histogramParad = { 0 };
	histogramBayer = { };

	void *parameters = NULL;
	if (options.Histogram.HistogramType == FAST_HISTOGRAM_PARADE) {
		histogramParad.stride = options.Histogram.ColumnStride;
		parameters = &histogramParad;
	} else if (options.Histogram.HistogramType == FAST_HISTOGRAM_BAYER ||
		options.Histogram.HistogramType == FAST_HISTOGRAM_BAYER_G1G2
	) {
		histogramBayer.bayerPattern = options.Histogram.BayerPattern;
		parameters = &histogramBayer;
	} 

	CHECK_FAST(fastHistogramCreate(
		&hHistogram,

		options.Histogram.HistogramType,
		parameters,
		options.Histogram.BinCount,

		options.MaxWidth,
		options.MaxHeight,

		srcBuffer
	));

	unsigned int histCount = GetHistogramCount(
		options.Histogram.HistogramType,
		surfaceFmt,
		options.Histogram.RoiWidth,
		options.Histogram.ColumnStride
	);

	cudaEventCreate(&event);

	FastAllocator alloc;
	CHECK_FAST_ALLOCATION(histCPU.reset((unsigned int *)alloc.allocate(histCount * options.Histogram.BinCount * sizeof(unsigned int))));

	size_t requestedMemSpace = 0;
	size_t tmp;
	CHECK_FAST(fastImportFromHostGetAllocatedGpuMemorySize(hImportFromHost, &tmp));
	requestedMemSpace += tmp;
	CHECK_FAST(fastHistogramGetAllocatedGpuMemorySize(hHistogram, &tmp));
	requestedMemSpace += tmp;
	printf("\nRequested GPU memory space: %.2f MB\n\n", requestedMemSpace / (1024.0 * 1024.0));

	return FAST_OK;
}

fastStatus_t Histogram::Close() const {
	cudaEventDestroy(event);
	CHECK_FAST(fastHistogramDestroy(hHistogram));
	CHECK_FAST(fastImportFromHostDestroy(hImportFromHost));
	return FAST_OK;
}

void Histogram::SaveParadeToFile(const char *fname) const {
	const unsigned int histCount = GetHistogramCount(
		options.Histogram.HistogramType,
		surfaceFmt,
		options.Histogram.RoiWidth,
		options.Histogram.ColumnStride
	);
	unsigned int histPerChannel = histCount / 3;
	FILE *fp = fopen(fname, "w");
	if (fp == NULL) {
		fprintf(stderr, "Can not create output file\n");
		return;
	}
	for (int channel = 0; channel < 3; channel++) {
		switch (channel) {
			case 0:
				fprintf(fp, "\nCHANNEL R\n\n");
				break;
			case 1:
				fprintf(fp, "\nCHANNEL G\n\n");
				break;
			case 2:
				fprintf(fp, "\nCHANNEL B\n\n");
				break;
			default:
				break;
		}
		for (unsigned column = 0; column < histPerChannel; column++) {
			fprintf(fp, "Column %d\n", column);
			for (int i = 0; i < options.Histogram.BinCount; i++) {
				fprintf(fp, "%d: %d\n", i, histCPU.get()[channel * histPerChannel * options.Histogram.BinCount + column + i * histPerChannel]);
			}
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
}

void Histogram::SaveHistogramToFile(const char *fname) const {
	unsigned int histCount = GetHistogramCount(
		options.Histogram.HistogramType,
		surfaceFmt,
		options.Histogram.RoiWidth,
		options.Histogram.ColumnStride
	);
	FILE *fp = fopen(fname, "w");
	if (fp == NULL) {
		fprintf(stderr, "Can not create output file\n");
		return;
	}
	for (unsigned hist = 0; hist < histCount; hist++) {
		fprintf(fp, "Histogram %d\n", hist + 1);
		for (int i = 0; i < options.Histogram.BinCount; i++) {
			fprintf(fp, "%d: %d\n", i, histCPU.get()[options.Histogram.BinCount * hist + i]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

fastStatus_t Histogram::Calculate(std::list< Image<FastAllocator> > &image) const {
	float fullTime = 0.;
	float elapsedTimeGpu = 0.;

	fastGpuTimerHandle_t hostToDeviceTimer = NULL;
	fastGpuTimerHandle_t histogramTimer = NULL;
	if (info) {
		fastGpuTimerCreate(&hostToDeviceTimer);
		fastGpuTimerCreate(&histogramTimer);
	}

	for (auto i = image.begin(); i != image.end(); ++i) {
		Image<FastAllocator> &img = *i;

		fprintf(stdout, "Input image: %s\nImage size: %dx%d pixels\n\n", img.inputFileName.c_str(), img.w, img.h);
		if (img.w > options.MaxWidth || img.h > options.MaxHeight) {
			fprintf(stderr, "Unsupported image size\n");
			continue;
		}

		if ((static_cast<unsigned>(options.Histogram.RoiLeftTopX + options.Histogram.RoiWidth) > img.w) || (static_cast<unsigned>(options.Histogram.RoiLeftTopY + options.Histogram.RoiHeight) > img.h)) {
			fprintf(stderr, "Incorrect coordinates for start point of analysed image area\n");
			return FAST_INTERNAL_ERROR;
		}

		if (info) {
			fastGpuTimerStart(hostToDeviceTimer);
		}
		CHECK_FAST(fastImportFromHostCopy(
			hImportFromHost,
			img.data.get(),
			img.w,
			img.wPitch,
			img.h
		));
		if (info) {
			fastGpuTimerStop(hostToDeviceTimer);
			fastGpuTimerGetTime(hostToDeviceTimer, &elapsedTimeGpu);

			fullTime += elapsedTimeGpu;
			printf("Host-to-device transfer = %.2f ms\n\n", elapsedTimeGpu);

			fastGpuTimerStart(histogramTimer);
		}

		CHECK_FAST(fastHistogramCalculate(
			hHistogram,
			NULL,
			options.Histogram.RoiLeftTopX,
			options.Histogram.RoiLeftTopY,
			options.Histogram.RoiWidth,
			options.Histogram.RoiHeight,
			histCPU.get()
		));
		

		if (cudaEventRecord(event) != cudaSuccess) {
			fprintf(stderr, "Synchronization failed\n");
			return FAST_EXECUTION_FAILURE;
		}

		if (cudaEventSynchronize(event) != cudaSuccess) {
			fprintf(stderr, "Synchronization failed\n");
			return FAST_EXECUTION_FAILURE;
		}

		if (info) {
			fastGpuTimerStop(histogramTimer);
			fastGpuTimerGetTime(histogramTimer, &elapsedTimeGpu);
			printf("Histogram and device-to-host copy time = %.2f ms\n\n", elapsedTimeGpu);

			fullTime += elapsedTimeGpu;
		}

		printf("Output file: %s\n\n", img.outputFileName.c_str());
		if (options.Histogram.HistogramType == FAST_HISTOGRAM_PARADE) {
			SaveParadeToFile(img.outputFileName.c_str());
		} else {
			SaveHistogramToFile(img.outputFileName.c_str());
		}
	}

	if (info) {
		printf("Total for all images = %.2f ms\n", fullTime);
		fastGpuTimerDestroy(hostToDeviceTimer);
		fastGpuTimerDestroy(histogramTimer);
	}

	return FAST_OK;
}
