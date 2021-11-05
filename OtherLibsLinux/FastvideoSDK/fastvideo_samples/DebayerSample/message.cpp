/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "DebayerSample project";
const char *helpProject =
	"The DebayerSample demonstrates demosaic functionality.\n"\
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in PGM/TIFF formats.\n" \
	" -o <file> - output file or mask in BMP/PPM/TIFF formats.\n" \
	" -if <folder + mask> - folder path and mask for input file. Extension should be PGM or TIFF.\n" \
	" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
	" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
	" -type <demosaic type> - demosaic type (MG, DFPD, HQLI).\n"\
	" -pattern <pattern> - bayer pattern (RGGB, BGGR, GBRG, GRBG).\n"\
	" -w <width> - input file width in pixels. RAW files only.\n" \
	" -h <height> - input file height. RAW files only.\n" \
	" -bits <bits count> - bits count per pixel (12 or 8 bits). RAW files only.\n" \
	" -black_shift <unsigned byte> - black shift values.\n"\
	" -R <float> -G1 <float> -G2 <float> -B <float> - white balance correction coefficients for R, G1, G2, B channels respectively.\n"\
	" -matrixA <file> - file with intensity correction values for each image pixel. File is in PFM format.\n"\
	" -matrixB <file> - file with black shift values for each image pixel. File is in PGM format.\n"\
	" -lut <file> - text file with LUT table for pixel transformation.\n" \
	" -repeat <value> - how many times to repeat encoding for one image.\n" \
	" -d <device ID> - GPU device ID. Default is 0.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -log <file> - enable log file.\n"\
	" -async - enable async mode to overlap processing and file read/write.\n"\
	" -thread <value> - number of processing threads in async mode. Default value is 1.\n"
	" -threadR <value> - number of file reader threads in async mode. Default value is 1.\n"
	" -threadW <value> - number of file writer threads in async mode. Default value is 1.\n"
	" -b <value> - batch size in async mode. Default value is 1.\n" \
	"\n" \
	"Performance Note: Application measures time of GPU kernels only. This time does not include neither times between kernels, nor transfer times. "\
	"When -thread parameter is set (1, 2, etc.), then application measures CPU time starting from host-to-device at the beginning of the pipeline "\
	"to device-to-host transfer at the end of the pipeline.\n"\
	"\n" \
	"Command line for Debayer:\n" \
	"DebayerSample.exe -i <input image> -o <output image> -p <pattern> -type <demosaic type> -d <device ID> -info \n"\
	"   -if <folder + mask> -maxWidth <unsigned int> -maxHeight <unsigned int> \n"\
	"	-black_shift <unsigned byte> -wb_R <float> -wb_G <float> -wb_B <float> -matrixA <file> -matrixB <file>\n" \
	"	-lut <file> -w <width> -h <height> -bits <bits count>\n" \
	"\n" \
	"Example of command line for color restore:\n" \
	"DebayerSample.exe -i test.pgm -o final.bmp -pattern RGGB -d 0 -info\n"\
	 "\n" \
	"Example of command line for color restore and multi thread mode:\n" \
	"DebayerSample.exe -i test.pgm -o final.bmp -pattern RGGB -d 0  -thread 2 -info\n"\
	"\n" \
	"Example of command line for color restore with pixel transform:\n" \
	"DebayerSample.exe -i test.pgm -o final.bmp -pattern RGGB -black_shift 0.0 -R 1.0 -G1 1.0 -G2 1.0 -B 1.0 -matrixA intensity.pfm -matrixB noise.pgm\n"\
	 "\n" \
	"Example of command line for color restore of 12 bit RAW image:\n" \
	"DebayerSample.exe -i Calendar.1920x1080.RGGB.raw -w 1920 -h 1080 -bits 12 -o final.bmp -type DFPD -pattern RGGB -info\n";
