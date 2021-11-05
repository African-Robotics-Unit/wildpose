/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "BayerCompressionSample project";
const char *helpProject =
	"The BayerCompressionSample demonstrates debayer merger and splitter process for single file and folder.\n"
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in pgm format for splitting and encoding and in JPG format for decoding and merging.\n" \
	" -if <folder + mask> - folder path and mask for input file. Extension should be JPG.\n" \
	" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
	" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
	" -o <file> - output file name in JPG format for decoding and merging and in ppm format for splitting and encoding. Or output folder path with mask for multiple files.\n" \
	" -type <demosaic type> - demosaic type (DFPD, HQLI).\n"\
	" -pattern <pattern> - bayer pattern (RGGB, BGGR, GBRG, GRBG).\n"\
	" -q <unsigned int> - JPEG encoder quality.\n" \
	" -s <subsampling> - subsampling 444, 422 or 420 for color images. Default is 444.\n" \
	" -d <device ID> - GPU device ID. Default is 0.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -log <file> - enable log file.\n"\
	"\n" \
	"Command line for BayerCompressionSample:\n" \
	"BayerCompressionSample.exe -i <input image> -o <output image> -pattern <pattern> -q <unsigned int> -s <subsampling> -r <restart interval> -d <device ID> -info\n"\
	"\n" \
	"Example of command line for BayerCompressionSample:\n" \
	"BayerCompressionSample.exe -i test2048.pgm -i final1024.jpg -q 85 -pattern RGGB -q 90 -r 4 -s 444 -info\n"\
	"\n" \
	"Example of command line for BayerCompressionSample with folder processing:\n" \
	"BayerCompressionSample.exe -if ./Images/*.pgm -o ./*.512.jpg -pattern RGGB -maxWidth 2048 -maxHeight 1565 -q 90 -r 4 -s 444 -info";
