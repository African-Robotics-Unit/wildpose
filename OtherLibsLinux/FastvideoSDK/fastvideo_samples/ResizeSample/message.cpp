/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "ResizeSample project";
const char *helpProject =
	"The ComponentsSample demonstrates base color correction and lut for color image.\n"
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in BMP/PGM/PPM/TIFF format.\n" \
	" -if <folder + mask> -  - folder path and mask for input file in BMP/PGM/PPM/TIFF formats.\n" \
	" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
	" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
	" -o <file> - output file in PPM format.\n" \
	" -outputWidth <width> - set output width.\n" \
	" -outputHeight <height> - set output height.\n" \
	" -d <device ID> - GPU device ID. Default is 0.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -background <R,G,B> - enable resize with padding and set background color for padding. "\
	" -log <file> - enable log file.\n"\
	"\n" \
	"Command of command line with fixed scale:\n" \
	"ResizeSample.exe -i <input image> -o <output image> -outputWidth <width> -d <device ID> -info\n" \
	"\n" \
	"Example of command line with custom scaled width and height:\n" \
	"ResizeSample.exe -i test.ppm -o final.ppm -outputWidth <width> -outputHeight <height> -info\n";
