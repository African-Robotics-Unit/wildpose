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

static const char *helpUrl = "www.fastcompression.com\n";
#ifndef __GNUC__
static const char *helpCompiledFor = "Compiled for Windows-7/8/10 [x64]\n";
#else
static const char *helpCompiledFor = "Compiled for Linux\n";
#endif
static const char *helpCommonInfo =
	"\n" \
	"This software is prepared for non-commercial use only. It is free for personal and educational (including non-profit organization) use. Distribution of this software without any permission from Fastvideo is NOT allowed. NO warranty and responsibility is provided by the authors for the consequences of using it.\n" \
	"\n" \
	"Hardware Requirements\n" \
	" - NVIDIA GeForce GPU 600, 700, 800, 900, 1000 or 2000 series with Compute Capability >= 3.0, or NVIDIA Quadro / Tesla, NVIDIA drivers 436.15 or later"
	"For the latest NVIDIA drivers visit http://www.nvidia.com/Download/index.aspx\n" \
	"\n";
static const char *sdkName = " for Fastvideo Image and Video Processing SDK - Copyright 2012 - 2021 Fastvideo\n";

static const char *sdkNameAndVersion = " for CUDA, based on Fastvideo SDK v.0.16.4.0 - Copyright 2012-2021 Fastvideo\n";

extern const char *projectName;
extern const char *helpProject;

void helpPrint(void) {
	printf("%s", projectName);
	printf("%s", sdkName);
	printf("%s", helpUrl);
	printf("%s", helpCompiledFor);
	printf("%s", helpCommonInfo);
	printf("%s", helpProject);
}

void helpDemoPrint(void) {
	printf("%s", projectName);
	printf("%s", sdkNameAndVersion);
	printf("%s", helpUrl);
	printf("%s", helpCompiledFor);
	printf("%s", helpCommonInfo);
	printf("%s", helpProject);
}