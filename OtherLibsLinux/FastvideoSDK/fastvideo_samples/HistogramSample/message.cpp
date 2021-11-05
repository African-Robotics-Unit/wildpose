/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "HistogramSample project";
const char *helpProject =
	"The HistogramSample demonstrates histogram component.\n"
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in BMP/PGM/PPM/TIFF formats.\n" \
	" -if <folder + mask> - folder path and mask for input file in BMP/PGM/PPM/TIFF formats.\n" \
	" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
	" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
	" -o <file> - output file in txt format.\n" \
	" -sX <unsigned int> - ofset X of region of interest.\n" \
	" -sY <unsigned int> - ofset Y of region of interest.\n" \
	" -roiWidth <unsigned int> - set width of region of interest.\n" \
	" -roiHeight <unsigned int> - set height of region of interest.\n" \
	" -bins <unsigned int> - bin count in histogram. Default 256.\n" \
	" -htype <unsigned int> - histogram type.\n" \
	"     0 - simple histogram.\n" \
	"     1 - bayer histogram.\n" \
	"     2 - bayer histogram with separate G1 G2.\n" \
	"     3 - parade.\n" \
	" -pattern {RGGB, GRBG, GBRG, BGGR} - bayer pattern.\n" \
	" -cstride <unsigned int> - column stride for parade.\n" \
	" -d <device ID> - GPU device ID. Default is 0.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -log <file> - enable log file.\n"\
	"\n" \
	"Example of command line for simple histogram:\n" \
	"HistogramSample.exe -i <input image> -o <output text> -htype 0 -bins 256 -sX 0 -sX 0 -roiWidth <width> -roiHeight <height> -info\n" \
	"\n" \
	"Example of command line for bayer histogram:\n" \
	"HistogramSample.exe -i <input image> -o <output text> -htype 1 -pattern RGGB -bins 256 -sX 0 -sX 0 -roiWidth <width> -roiHeight <height> -info\n";
