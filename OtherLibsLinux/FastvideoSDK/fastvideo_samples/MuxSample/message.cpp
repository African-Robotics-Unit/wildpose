/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "MuxSample project";
const char *helpProject =
	"The MuxSample demonstrates Multiplexer components.\n"\
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in PGM format.\n" \
	" -if <folder + mask> - folder path and mask for input file. 12 bit PPM format supported.\n" \
	" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
	" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
	" -o <file> - output file name in PPM format.\n" \
	" -type <demosaic type> - demosaic type (DFPD, HQLI).\n"\
	" -pattern <pattern> - bayer pattern (RGGB, BGGR, GBRG, GRBG).\n"\
	" -matrixA <file> - file with intensity correction values for each image pixel. File is in PFM format.\n"\
	" -matrixB <file> - file with black shift values for each image pixel. File is in PGM format.\n"\
	" -d <device ID> - GPU device ID. Default is 0.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -log <file> - enable log file.\n"\
	"\n" \
	"Command line for MuxSample:\n" \
	"MuxSample.exe -i <input image> -o <output image> -type <type> -pattern <pattern> -matrixA <maxtrix A> -d <device ID> -info\n" \
	"\n" \
	"Example of command line:\n" \
	"MuxSample.exe -i test.pgm -o final.ppm -pattern RGGB -d 0 -info\n";
