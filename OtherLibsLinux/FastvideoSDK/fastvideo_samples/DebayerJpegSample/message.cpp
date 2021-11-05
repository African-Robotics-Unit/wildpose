/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "DebayerJpegSample project";
const char *helpProject =
"The DebayerJpegSample demonstrates Debayer with JPEG Encoder integration.\n"
"\n" \
"Supported parameters\n" \
" -i <file> - input file in PGM or TIFF formats.\n" \
" -if <folder + mask> - folder path and mask for input file. Extension should be PGM or TIFF.\n" \
" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
" -o <file> - output file or mask in JPEG format.\n" \
" -q <quality> - quality setting according to JPEG standard. Default is 75.\n" \
" -s <subsampling> - subsampling 444, 422 or 420 for color images. Default is 444.\n" \
" -type <demosaic type> - demosaic type (DFPD, HQLI).\n"\
" -pattern <pattern> - bayer pattern (RGGB, BGGR, GBRG, GRBG).\n"\
" -d <device ID> - GPU device ID. Default is 0.\n" \
" -info - time / performance output is on. Default is off.\n" \
" -log <file> - enable log file.\n"\
"\n" \
"Command line for DebayerJpegSample:\n" \
"DebayerJpegSample.exe -i <input image> -o <output image> -s <subsampling> -q <quality> -r <restart interval> \n" \
"   -if <folder + mask> -maxWidth <unsigned int> -maxHeight <unsigned int> \n"\
"   -p <pattern> -type <demosaic type> -d <device ID> -info\n"\
"\n" \
"Example of command line for color restore:\n" \
"DebayerJpegSample.exe -i test.pgm -o final.jpg -s 444 -q 75 -r 8 -d 0 -pattern RGGB -type DFPD -info\n"\
"\n";
	