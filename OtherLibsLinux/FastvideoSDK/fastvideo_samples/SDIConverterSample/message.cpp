/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "SDIConverterSample project";
const char *helpProject =
	"The SDIConverterSample demonstrates file import/export in SDI format.\n"
	"\n" \
	"List of supported format: \n"\
	"\t8 bits depth format:\n" \
	"\t\t420: YV12_601_FR, YV12_601, YV12_709, NV12_601_FR, NV12_601, NV12_709;\n"\
	"\t\t422: CbYCr422_709, CbYCr422_601_FR, CbYCr422_601, CrYCb422_709, CrYCb422_601, CrYCb422_601_FR;\n"\
	"\t\t444: YCbCr444_709, YCbCr444_601, YCbCr444_601_FR;\n"\
	"\t\tRGBA;\n"\
	"\t10 bits depth format:\n" \
	"\t\t420: P010_601_FR, P010_601, P010_709, YCbCr420_10_601_FR, YCbCr420_10_601, YCbCr420_10_709;\n"\
	"\t\t444: YCbCr444_10_709, YCbCr444_10_601, YCbCr444_10_601_FR;\n"\
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in SDI/PPM/TIFF formats.\n" \
	" -o <file> - output file in SDI/PPM/TIFF formats.\n" \
	" -export - define export mode (default is import).\n" \
	" -width <unsigned int> - source image width (import mode).\n" \
	" -height <unsigned int> - source image height (import mode).\n" \
	" -format {format} - file format.\n" \
	" -d <device ID> - GPU device ID. Default is 0.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -log <file> - enable log file.\n"\
	"\n" \
	"Command line for SDIConverterSample:\n" \
	"SDIConverterSample.exe -i <input image> -o <output image> -format <format> -width <width> -height <height> -d <device ID> -info\n" \
	"\n" \
	"Command line sample for import:\n" \
	"SDIConverterSample.exe -i <input image> -o <output image> -format CbYCr422_709 -width 1920 -height 1080 -d <device ID> -info\n" \
	"\n" \
	"Command line sample for export:\n" \
	"SDIConverterSample.exe -export -i <input image> -o <output image> -format CbYCr422_709 -width 1920 -height 1080 -d <device ID> -info\n";
