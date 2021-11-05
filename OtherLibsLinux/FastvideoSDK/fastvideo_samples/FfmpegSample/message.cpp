/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "FfmpegSample project";
const char *helpProject =
	"The FfmpegSample demonstrates image compression to Motion JPEG and decompression from Motion JPEG.\n"
	"\n" \
	"Supported parameters\n" \
	" -i <file> - Motion JPEG file in avi container.\n" \
	" -if <folder> - folder with files in JPEG format.\n" \
	" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
	" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
	" -o <file> - output file name of avi file for encoder. Or output folder path with mask for multiple files for decoder.\n" \
	" -q <quality> - quality setting according to JPEG standard. Default is 75.\n" \
	" -s <subsampling> - subsampling 444, 422 or 420 for color images. Default is 444.\n" \
	" -frameRate <unsigned int> - frame rate for motion jpeg file. Default value 24.\n" \
	" -frameRepeat <unsigned int> - how many times to repeat compression of just one image to create motion jpeg file.\n" \
	" -d <device ID> - GPU device ID. Default is 0.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -log <file> - enable log file.\n"\
	"\n" \
	"Command line for FfmpegSample:\n" \
	"FfmpegSample.exe -i <input file> -if <input files> -o <output file/files> -outputWidth <new width> -q <unsigned int> -s <subsampling> \n"\
	"-r <restart interval> -d <device ID> -maxWidth <unsigned int> -maxHeight <unsigned int> -frameRate <unsigned int> -frameRepeat <unsigned int> -info\n"\
	"\n" \
	"Example of encoder command line for FfmpegSample:\n" \
	"FfmpegSample.exe -if ./Images/*.bmp -o ffmpeg.avi -frameRepeat 24 -MaxWidth 1920 -MaxHeight 1080 -info\n"\
	"\n" \
	"Example of decoder command line for FfmpegSample:\n" \
	"FfmpegSample.exe -i ffmpeg.avi -o ./Images/*.ppm -info";
