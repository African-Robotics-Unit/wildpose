/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "helper_bytestream.hpp"

fastStatus_t fvSaveBytestream(
	std::string fname,
	unsigned char* inputImg,
	size_t size,
	bool info
) {
	hostTimer_t timer = NULL;
	if (info) {
		timer = hostTimerCreate();
		hostTimerStart(timer);
	}

	std::ofstream output(fname.c_str(), std::ofstream::binary);
	if (output.is_open()) {
		output.write((char*)inputImg, size);
		output.close();
	}
	else {
		return FAST_IO_ERROR;
	}

	if (info) {
		const double loadTime = hostTimerEnd(timer);
		printf("JFIF image write time = %.2f ms\n\n", loadTime * 1000.0);
		hostTimerDestroy(timer);
	}

	return FAST_OK;
}