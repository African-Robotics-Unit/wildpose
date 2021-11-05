/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "fastvideo_sdk.h"

#include "BaseOptions.h"

#include "Help.h"
#include "GPU.hpp"
#include "DecodeError.hpp"

#include "LicenseInfoPrinter.hpp"
#include "checks.h"

bool sampleInit(int argc, char *argv[], BaseOptions &options) {
	if (!options.Parse(argc, argv)) {
		helpPrint();
		return false;
	}

	if (options.Help) {
		helpPrint();
		return false;
	}

	if (!LicenseInfoPrinter()) {
		return false;
	}

	if (!DecodeError(setGpuDeviceId(options.DeviceId))) {
		return false;
	}

	if (options.LogFile != nullptr) {
		CHECK_FAST(fastTraceCreate(options.LogFile));
		CHECK_FAST(fastTraceEnableLUTDump(true));
	}
	
	return true;
}

bool sampleDestroy(BaseOptions &options) {
	if (options.LogFile != nullptr) {
		CHECK_FAST(fastTraceClose());
	}

	return true;
}
