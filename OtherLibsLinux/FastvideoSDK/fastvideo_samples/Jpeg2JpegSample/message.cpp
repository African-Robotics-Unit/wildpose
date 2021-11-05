/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "Jpeg2JpegSample project";
const char *helpProject =
	"The Jpeg2JpegSample demonstrates JPEG to JPEG resize pipeline for single file and folder.\n"
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in JPG format.\n" \
	" -o <file> - output file or mask in JPG format.\n" \
	" -if <folder + mask> - folder path and mask for input file. Extension should be JPG.\n" \
	" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
	" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
	" -outputWidth <new width> - new image width in pixels. Default is the same as the input.\n" \
	" -q <unsigned int> - JPEG encoder quality.\n" \
	" -s <subsampling> - subsampling 444, 422 or 420 for color images. Default is 444.\n" \
	" -sharp_before <sigma> - sharpen correction before resize algorithm enabled. Default is off.\n" \
	" -sharp_after <sigma> - sharpen correction after resize algorithm enabled. Default is off.\n" \
	" -crop <width x height + offsetX + offsetY> - crop paremeters. It defines ROI size and left corner offset.\n" \
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
	"Command line for Jpeg2JpegSample:\n" \
	"Jpeg2JpegSample.exe -i <input image> -o <output image> -outputWidth <new width> -q <unsigned int> -s <subsampling> -d <device ID>  -thread <value> -repeat <value> -info\n"\
	"\n" \
	"Example of command line for Jpeg2JpegSample:\n" \
	"Jpeg2JpegSample.exe -i test2048.jpg -o final1024.jpg -outputWidth 1024 -q 90 -s 444 -info\n"\
	"\n" \
	"Example of command line for Jpeg2JpegSample in multi thread mode:\n" \
	"Jpeg2JpegSample.exe -i test2048.jpg -o final1024.jpg -outputWidth 1024 -q 90 -s 444 -async -thread 2 -threadR 1 -threadW 1 -b 2\n"\
	"\n" \
	"Example of command line for Jpeg2JpegSample with folder processing:\n" \
	"Jpeg2JpegSample.exe -if ./Images/*.jpg -o ./*.512.jpg -outputWidth 512 -maxWidth 2048 -maxHeight 1565 -q 90 -r 4 -s 444 -info";
