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

#include "helper_quant_table.hpp"
#include "helper_common.h"
#include "helper_jpeg/helper_jpeg.hpp"

fastStatus_t fvLoadQuantTable(const char *file, fastJpegQuantState_t& quantState) {
	FILE *fp = NULL;

	quantState = { 0 };
	if (FOPEN_FAIL(FOPEN(fp, file, "r")))
		return FAST_IO_ERROR;

	for (int i = 0; i < 2; i++) {
		unsigned j = 0;
		while (j < DCT_SIZE * DCT_SIZE && !feof(fp)) {
			int value;
			fscanf(fp, "%d", &value);
			quantState.table[i].data[j] = value;
			j++;
		}
	}
	fclose(fp);

	return FAST_OK;
}