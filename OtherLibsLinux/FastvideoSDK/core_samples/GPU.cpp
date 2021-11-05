/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <cstdio>
#include <new>

#include <cuda_runtime.h>
#include <nvml.h>

#include "fastvideo_sdk.h"

#include "checks.h"
#include <cstring>

int getDeviceProperties(cudaDeviceProp *prop, int deviceId) {
	const cudaError_t error = cudaGetDeviceProperties(prop, deviceId);
	if (error != cudaSuccess) {
		fprintf(stderr, "%s\n", cudaGetErrorString(error));
		return -1;
	}
	return 0;
}

fastStatus_t bandwidthTest(float &bandwidthInMBs) {
	const size_t memSize = 100 * (1 << 20);	// 100 MB

	float elapsedTimeInMs = 0.0f;

	cudaEvent_t start, stop;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	//allocate host memory
	unsigned char *h_odata = NULL;
	CHECK_FAST_ALLOCATION(fastMalloc((void **)&h_odata, memSize));

	//initialize the memory
	for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++)
		h_odata[i] = (unsigned char)(i & 0xff);

	//allocate device memory
	unsigned char *d_idata;
	CHECK_CUDA(cudaMalloc((void **)&d_idata, memSize));
	CHECK_CUDA(cudaEventRecord(start, 0));

	//copy host memory to device memory
	CHECK_CUDA(cudaMemcpy(d_idata, h_odata, memSize, cudaMemcpyHostToDevice));

	CHECK_CUDA(cudaEventRecord(stop, 0));
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

	//calculate bandwidth in MB/s
	bandwidthInMBs = (1e3f * memSize) / (elapsedTimeInMs * (float)(1 << 20));

	//clean up memory
	CHECK_CUDA(cudaEventDestroy(stop));
	CHECK_CUDA(cudaEventDestroy(start));
	CHECK_FAST(fastFree(h_odata));
	CHECK_CUDA(cudaFree(d_idata));

	return FAST_OK;
}

fastStatus_t GetNvmlMemoryInfo(int cudaDeviceId, size_t *freeMemory, size_t* totalMemory)
{

#ifndef NVML_DISABLE
	nvmlReturn_t resultNvml = nvmlInit();
	if (resultNvml != NVML_SUCCESS) {
		printf("Failed to initialize NVML: %s\n", nvmlErrorString(resultNvml));
		return FAST_INVALID_DEVICE;
	}
	const int idLength = 13;
	char cudaPciBusId[idLength + 1]; // [domain]:[bus]:[device].[function] where domain, bus, device, and function are all hexadecimal values.
	cudaError_t resultCuda = cudaDeviceGetPCIBusId(cudaPciBusId, idLength, cudaDeviceId);
	if (resultCuda != cudaSuccess) 	
		return FAST_INVALID_DEVICE;

	unsigned device_count = 0;
	resultNvml = nvmlDeviceGetCount(&device_count);
	if (resultNvml != NVML_SUCCESS || (int)device_count <= cudaDeviceId) {
		return FAST_INVALID_DEVICE;
	}
	
	nvmlDevice_t nvmlDevice = nullptr;
	nvmlPciInfo_t pciInfo;
	unsigned nvmlDeviceId = 0;
	for (; nvmlDeviceId < device_count; nvmlDeviceId++)
	{
		resultNvml = nvmlDeviceGetHandleByIndex(nvmlDeviceId, &nvmlDevice);
		if (resultNvml != NVML_SUCCESS) 
			return FAST_INVALID_DEVICE;
		resultNvml = nvmlDeviceGetPciInfo(nvmlDevice, &pciInfo);
		if (resultNvml != NVML_SUCCESS) 
			return FAST_INVALID_DEVICE;
		if (strncmp(cudaPciBusId, pciInfo.busIdLegacy, idLength) == 0)
			break;
	}

	nvmlMemory_t memInfo;
	resultNvml = nvmlDeviceGetMemoryInfo(nvmlDevice, &memInfo);
	if (resultNvml != NVML_SUCCESS) {
		printf("Failed to get memory info for device %u: %s\n", nvmlDeviceId, nvmlErrorString(resultNvml));
		nvmlShutdown();
		return FAST_INVALID_DEVICE;
	}
	if (freeMemory != nullptr)
		*freeMemory = memInfo.free;
	if (totalMemory != nullptr)
		*totalMemory = memInfo.total;
#else
	cudaError_t res = cudaMemGetInfo(freeMemory, totalMemory);
	if (res != cudaSuccess)
		return FAST_INVALID_DEVICE;
#endif
	return FAST_OK;
}

fastStatus_t setGpuDeviceId(int deviceId) {
	int deviceCount;
	float bandwidth = 0.0f;

	cudaError_t error_code = cudaGetDeviceCount(&deviceCount);
	switch (error_code) {
		case cudaSuccess:
			if (deviceId > deviceCount - 1) {
				fprintf(stderr, "Device id %d is not found. Set to default\n", deviceId);
				deviceId = 0;
			}

			cudaDeviceProp prop;
			if (getDeviceProperties(&prop, deviceId) < 0)
				return FAST_INVALID_DEVICE;

			if (prop.major < 2) {
				fprintf(stderr, "GPU is not supported. Program runs on GPU with Compute Compatibility 2.0 or higher\n");
				return FAST_INVALID_DEVICE;
			}

			CHECK_FAST(fastInit(1U << deviceId, false));
			printf("Processing unit: %s (device id = %d)\n", prop.name, deviceId);

			{
				size_t freeMemory, totalMemory;
				GetNvmlMemoryInfo(deviceId, &freeMemory, &totalMemory);
				printf("Available GPU memory size: %.2f GB\n", freeMemory / (1024.0 * 1024.0 * 1024.0));

			}

			if (prop.tccDriver) {
				printf("TCC driver mode enabled\n");
			}

			CHECK_FAST(bandwidthTest(bandwidth));
			printf("PCI-Express bandwidth test (host to device): %d MByte/s\n\n", (unsigned)bandwidth);
			break;
		case cudaErrorNoDevice:
		case cudaErrorInsufficientDriver:
		default:
			fprintf(stderr, "CUDA capability device or driver not found\n");
			return FAST_INVALID_DEVICE;
	}

	return FAST_OK;
}
