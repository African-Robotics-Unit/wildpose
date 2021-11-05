/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "NppSample project";
const char *helpProject =
"The NppSample demonstrates NPP components of SDK.\n"
"\n" \
"Supported parameters\n" \
" -i <file> - input file in BMP/PGM/PPM/TIFF formats.\n" \
" -if <folder + mask> - folder path and mask for input file in BMP/PGM/PPM/TIFF formats.\n" \
" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
" -o <file> - output file in PPM format.\n" \
" -gauss - apply gauss filter.\n" \
" -unsharp - apply unsharp filter.\n" \
" -sigma <float> - (for -gauss and -unsharp) parameter defines gauss kernel.\n" \
" -radius <float> - (for -gauss and -unsharp) radius of gaussian kernel.\n" \
" -amount <float> - (for -unsharp) controls how much contrast is added at the edges.\n" \
" -threshold <float> -(for -unsharp) controls the minimum brightness change that will be sharpened.\n" \
" -resize - apply NPP resize algorithm.\n" \
" -rotate - apply NPP rotate algorithm.\n"\
" -resizedWidth <int> - (for -resize) resized width.\n" \
" -shift <float> - (for -resize) shift between source and destination grids.\n"\
" -angle <float> - (for -rotate) rotation angle in degrees.\n" \
" -interpolationType <type> - (for -rotate, -resize, -remap,  -remap3) type of interpolation function. There are linear, cubic, bspline, catmullrom, b05c03, super, lanczos.\n" \
" -remap - apply NPP remap algorithm.\n"\
" -remap3 - apply NPP remap3 algorithm.\n"\
" -rotate90 - (for -remap and remap3) operation emulated by remap function. Remap without parameters emulates horizontal flip.\n"\
" -R <int>, -G <int>, -B <int>  - (for -remap and remap3) background pixel for destination image.\n"\
" -d <device ID> - GPU device ID. Default is 0.\n" \
" -info - time / performance output is on. Default is off.\n" \
" -log <file> - enable log file.\n"\
"\n" \
"Command line for NppSample:\n" \
"NppSample.exe -i <input image> -o <output image>  -d <device ID> -info -log <file>\n"\
"    -if <folder + mask> -maxWidth <unsigned int> -maxHeight <unsigned int>  \n"\
"    -gauss -unsharp -sigma <float> -radius <float> -amount <float> -threshold <float>\n"\
"    -resize -resizedWidth <int> -shift <float> -interpolationType <type>\n"\
"    -rotate -angle <float> -interpolationType <type>\n"\
"    -remap -remap3 -rotate90 -R <int>, -G <int>, -B <int> \n"\
"\n" \
"Command line for gauss filter:\n" \
"NppSample.exe -gauss -i input.ppm -o output.ppm -sigma 0.95 -radius 1.0 -info\n" \
"\n" \
"Command line for unsharp filter:\n" \
"NppSample.exe -unsharp -i input.ppm -o output.ppm -sigma 0.95 -amount 0.95 -threshold 0.1 -info\n"
"\n" \
"Command line for resize:\n" \
"NppSample.exe -resize -i input.ppm -o output.ppm -resizedWidth 1000 -shift 0.5 -interpolationType cubic -info\n"\
"\n" \
"Command line for rotate:\n" \
"NppSample.exe -rotate -i input.ppm -o output.ppm -angle 45 -interpolationType cubic -info\n"
"\n" \
"Command line for remap and remap3:\n" \
"NppSample.exe -remap -i input.ppm -o output.ppm -rotate90 -R 4095 -G 4095 -B 4095 -info\n";

