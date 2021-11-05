/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "RawImportSample project";
const char *helpProject =
	"The RawImportSample demonstrates file import in RAW format (XIMEA and PTG versions).\n"
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in raw format.\n" \
	" -o <file> - output file in PGM/PPM/BMP/TIFF formats.\n" \
	" -format <format> - file format (ximea12, ptg12).\n" \
	" -width <unsigned int> - source image width.\n" \
	" -height <unsigned int> - source image height.\n" \
	" -bits <unsigned int> - bits per channel (just 12 bits support).\n" \
	" -d <device ID> - GPU device ID. Default is 0.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -log <file> - enable log file.\n"\
	"\n" \
	"Command line for RawImportSample:\n" \
	"RawImportSample.exe -i <input image> -o <output image> -format <format> -width <width> -height <height> -bits 12 -d <device ID> -info\n" \
	"\n" \
	"Command line sample for import:\n" \
	"RawImportSample.exe -i <input image> -o <output image> -format ximea12 -width 1920 -height 1080 -bits 12 -d <device ID> -info\n";
