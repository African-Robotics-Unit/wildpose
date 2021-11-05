/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "JpegAsyncSample project";
const char *helpProject =
	"The JpegAsyncSample demonstrates JPEG encoder functionality in asynchronous mode.\n"\
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in BMP/PGM/PPM/TIFF format for encoding.\n" \
	" -if <folder + mask> - folder path and mask for input file. BMP/PGM/PPM/TIFF format for encoding.\n" \
	" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
	" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
	" -o <file> - output file name or mask in BMP/PGM/PPM/TIFF for decoding.\n" \
	" -q <quality> - quality setting according to JPEG standard. Default is 75.\n" \
	" -s <subsampling> - subsampling 444, 422 or 420 for color images. Default is 444.\n" \
	" -d <device ID> - GPU device ID. Default is 0.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -log <file> - enable log file.\n"\
	"\n" \
	"Command line for JpegAsyncSample:\n" \
	"JpegAsyncSample.exe -i <input image> -o <output image> -s <subsampling> -q <quality> -r <restart interval> -d <device ID> -info\n" \
	"\n" \
	"Example of command line for JPEG encoding:\n" \
	"JpegAsyncSample.exe -i test.bmp -o final.jpg -s 444 -q 75 -r 32 -d 0 -info\n";
