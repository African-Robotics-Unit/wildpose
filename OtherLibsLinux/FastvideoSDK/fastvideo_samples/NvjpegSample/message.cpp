/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "NvjpegSample project";
const char *helpProject =
"The NvjpegSample demonstrates JPEG decoder and encoder functionality.\n"\
"\n" \
"Supported parameters\n" \
" -i <file> - input file in BMP/PGM/PPM/TIFF format for encoding, JPG for decoding.\n" \
" -if <folder + mask> - folder path and mask for input file. BMP/PGM/PPM/TIFF format for encoding and JPG for decoding.\n" \
" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
" -o <file> - output file name or mask in JPG format for encoding, BMP/PGM/PPM/TIFF for decoding.\n" \
" -q <quality> - quality setting according to JPEG standard. Default is 75.\n" \
" -s <subsampling> - subsampling 444, 422 or 420 for color images. Default is 444.\n" \
" -noExif - generate jpeg started by APP0 marker (JFIF). Default format for jpeg is EXIF (started by APP1 marker).\n" \
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
"Performance Note: Application measures time of GPU kernels only. This time does not include neither times between kernels, nor transfer times.\n"\
"When -thread parameter is set (1, 2, etc.), then application measures CPU time starting from host-to-device at the beginning of the pipeline\n"\
"to device-to-host transfer at the end of the pipeline.\n\n"\
"\n" \
"Command line for NvjpegSample:\n" \
"NvjpegSample.exe -i <input image> -o <output image> -s <subsampling> -q <quality> \n" \
"    -repeat <value>  -d <device ID> -info -log <file\n" \
"\n" \
"Example of command line for JPEG encoding:\n" \
"NvjpegSample.exe -i test.bmp -o final.jpg -s 444 -q 75 -d 0 -info\n" \
"\n" \
"Example of command line for JPEG decoding:\n" \
"NvjpegSample.exe -i test.jpg -o final.bmp -d 0 -info\n"
"\n" \
"Example of command line for Multithreaded JPEG encoding:\n" \
"NvjpegSample.exe -i test.bmp -o final.jpg -s 444 -q 75 -thread 2 -d 0 -info\n" \
"\n" \
"Example of command line for Multithreaded JPEG decoding:\n" \
"NvjpegSample.exe -i test.jpg -o final.bmp -d 0 -async -thread 2 -threadR 1 -threadW 1 -b 2 -info\n"
"\n";
	
