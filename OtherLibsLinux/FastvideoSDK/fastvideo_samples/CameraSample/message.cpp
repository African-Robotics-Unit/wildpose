/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

const char *projectName = "CameraSample project";
const char *helpProject =
	"The CameraSample demonstrates basic camera pipeline finished by OpenGL visualisation. Processed images have been stored to motion JPEG format with FFMPEG. \n"\
	"List of demonstrated components:\n"\
	"\tMAD filter,\n" \
	"\tBase Color Correction filter,\n" \
	"\tDebayer,\n" \
	"\t8-bit 1D LUT filter,\n" \
	"\tJpeg Encoder,\n" \
	"\tMotion Jpeg Writer,\n" \
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in BMP/PGM/PPM format for encoding, JPG for decoding.\n" \
	" -if <folder + mask> - folder path and mask for input file. Extension should be  BMP/PGM/PPM.\n" \
	" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
	" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
	" -o <file> - output file name in AVI format.\n" \
	" -q <quality> - quality setting according to JPEG standard. Default is 75.\n" \
	" -s <subsampling> - subsampling 444, 422 or 420 for color images. Default is 444.\n" \
	" -colorCorrection {matrix} - color correction matrix 4x3.\n"\
	" -matrixA <file> - file with intensity correction values for each image pixel. File is in PFM format.\n"\
	" -matrixB <file> - file with black shift values for each image pixel. File is in PGM format.\n"\
	" -lut <file> - file with 8-bits 1D LUT. To convert 8 bit pixel to other 8 bit pixel.\n"\
	" -frameRepeat <value> - how many times repeat processing for one image.\n" \
	" -frameRate <unsigned int> - frame rate for motion jpeg file. Default value 24.\n" \
	" -d <device ID> - GPU device ID. Default is 0.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -log <file> - enable log file.\n"\
	"\n" \
	"Command line for CameraSample:\n" \
	"CameraSample.exe -i <input image> -o <output image> -s <subsampling> -q <quality> -r <restart interval> -d <device ID> -info \n" \
	"    -if <folder + mask> -maxWidth <unsigned int> -maxHeight <unsigned int> \n"\
	"    -colorCorrection {matrix} -matrixA <file> -matrixB <file> -lut <file> -frameRepeat <value> -log <file> \n" \
	"\n" \
	"Example of grayscale image encoding with pixel correction:\n" \
	"CameraSample.exe -if .\\Images\\*.pgm  -o final.avi -maxWidth 768 -maxHeight 512 -s 444 -q 75 -r 32 -colorCorrection {1,0,0,0,0,1,0,0,0,0,1,0} -matrixA intensity.pfm -matrixB noise.pgm -lut lut.txt -frameRepeat 24\n";

