/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "ImageConverterSample project";
const char *helpProject =
	"The ImageConverterSample converts 8 bits PGM/PPM to 12/16 bits PGM/PPM and vice versa. Also the ImageConverterSample converts 12/16 bits PGM to supported RAW and vice versa.\n"\
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in PGM/PMM/TIFF or RAW formats.\n" \
	" -o <file> - output file or mask in PGM/PMM/TIFF or RAW formats.\n" \
	" -w <width> - input file width in pixels. RAW files only.\n" \
	" -h <height> - input file height. RAW files only.\n" \
	" -bits <bits count> - bits count per pixel (12 or 8 bits).\n" \
	" -shift <shift bits count> - left shift bits for 8 to 12 bits conversion. Right shift bits for 12 to 8 bits conversion.\n" \
		"\n" \
	"Command line for Debayer:\n" \
	"ImageConverterSample.exe -i <input image> -o <output image> -w <width> -h <height> -bits <bits count> -shift <shift bits count> -info \n"\
	"\n" \
	"Example of command line to convert 8 bits PGM to 12 bit:\n" \
	"ImageConverterSample.exe -i test.pgm -o test.12.pgm -shift 4 -bits 12\n"\
	 "\n" \
	"Example of command line to convert 12 bits PGM to 8 bits:\n" \
	"ImageConverterSample.exe -i test.12.ppm -o test.ppm -shift 4 -bits 8\n"\
	 "\n" \
	"Example of command line to convert RAW to 12 bit PGM:\n" \
	"ImageConverterSample.exe -i Calendar.1920x1080.RGGB.raw -w 1920 -h 1080 -bits 12 -o test.12.pgm\n"\

	"Example of command line to convert 12 bit PGM to RAW:\n" \
	"ImageConverterSample.exe -i test.12.pgm -w 1920 -h 1080 -bits 12 -o out.raw\n";
