/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

const char *projectName = "DenoiseSample project";
const char *helpProject ="The DenoiseSample demonstrates Denoiser component.\n"
	"\n" \
	"Supported parameters:\n" \
	" -i <file> - input file in BMP/PGM/PPM/TIFF format.\n" \
	" -if <folder + mask> - folder path and mask for input file in BMP/PGM/PPM/TIFF formats.\n" \
	" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
	" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
	" -o <file> - output file in BMP/PGM/PPM/TIFF for decoding.\n" \
	" -CPU - use CPU for all operations (if not specified, use GPU when possible).\n" \
	" -a <name> - one of the algorithms (NOISE, MCT, DENOISE, TRANSFORM).\n" \
	" -n <number> - noise level for image contamination (e.g., 0.9;0.8;0.7 for three color channels).\n" \
	" -w <name> - one of the wavelets (HAAR, UHAAR, CDF53, CDF97, UCDF97).\n" \
	" -l <number> - number of resolution levels (in the range 1-12).\n" \
	" -f <name> - one of the thresholding functions (HARD, SOFT, GARROTE).\n" \
	" -t <number> - denoising threshold (e.g., 0.75;1;1 for three color channels).\n" \
	" -e <number> - enhance image if (value > 1.0), blur image if (0.0 < value < 1.0).\n" \
	" -shrink <name> - select one of the threshold computation methods: \n" \
	"                  VisuShrink, VisuShrink_2, SureShrink, BayesShrink, NormalShrink, NeighShrink.\n"
	" -post <name> - select the filter applied to wavelet coefficients after shrinkage (HANN3, HANN5).\n"
	" -d <device ID> - GPU device ID. Default is 0.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -log <file> - enable log file.\n"\
	"\n" \
	"Command line for Debayer:\n" \
	"DenoiseSample.exe -i source.ppm -o result.ppm -a <name>  -n <number>  -w <name> -l <number> -f <name> -t <number>  -e <number> \n"\
	"   -if <folder + mask> -maxWidth <unsigned int> -maxHeight <unsigned int> \n"\
	"	-shrink <name> -post <name>  -d <device ID> -info -log <file>"\
	"\n" \
	"Example of command line for denoising:\n" \
	"DenoiseSample.exe -i source.ppm -o result.ppm -a UHAAR -l 3 -t 3.0\n";
