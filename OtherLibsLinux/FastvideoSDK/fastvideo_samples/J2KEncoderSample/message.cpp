/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "J2kEncoderSample project";
const char *helpProject =
	"The J2kEncoderSample demonstrates JPEG2000 encoder functionality.\n"\
	"\n" \
	"There are three modes for J2K encoder sample application : single, batch, multithreaded batch.\n" \
	"\n" \
	"1. In the single mode every image is encoded one by one. It is the minimum latency mode with minimal memory consumption. Single mode is activated by default.\n" \
	"\n" \
	"2. In the batch mode, multiple images are encoded in one instance of the encoder simultaneously. Batch mode has better performance than single mode, but consumes much more GPU memory. Parameter -b defines batch size. Batch mode is activated when parameter -b is greater than one.\n" \
	"\n" \
	"In this sample you can easily test the batch mode starting repeated processing of one image via parameter -repeat. Faster encoding is achieved when -repeat value is multiple of -b.\n" \
	"\n" \
	"3. In the multithreaded batch mode we run two batch J2K encoders in separate threads. This allows us to overlap computations and data transfers to increase GPU utilization. This mode shows the maximum performance and quite high GPU memory usage. Setting -threads parameter to 2 activates this mode. Also you can increase number of threads over 2 to get even better performance. Use parameter -repeat to supply encoders with enough workload.\n" \
	"\n" \
	"Supported parameters\n" \
	" -i <file> - input file in BMP/PGM/PPM format.\n" \
	" -if <folder + mask> - folder path and mask for input file in BMP/PGM/PPM formats.\n" \
	" -o <file> - output file in JP2/J2K format.\n" \
	" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
	" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
	" -d <value> - GPU (device) index.\n" \
	" -a <name> - one of the encoding algorithms (REV - reversible/lossless, IRREV - irreversible/lossy).\n" \
	" -b <value> - batch size (maximum number of simultaneously processed images).\n" \
	" -c <value> - codeblock size: 16, 32 (default), 64.\n" \
	" -q <value> - quality (in the range 1-100). Specifies losses at quantization stage.\n" \
	" -l <value> - number of resolution levels (in the range 1-12).\n" \
	" -cr <value> - compression ratio. Specifies truncation of encoded codeblocks. Using this option enables PCRD stage.\n" \
	" -s <subsampling> - subsampling 444, 422 or 420 for color images. Default is 444.\n" \
	" -repeat <value> - how many times to repeat encoding for one image.\n" \
	" -discard - do not write the output to disk.\n" \
	" -noMCT - disable Multi-Component (YCbCr) transformation.\n" \
	" -noHeader - disable JP2 header generation.\n" \
	" -tileWidth <value> - tile width.\n" \
	" -tileHeight <value> - tile height.\n" \
	" -outputBitdepth <value> - defines output file bit depth. If parameter is not set, then output bit depth is taken from ppm/pgm file.\n" \
	" -overwriteSourceBitdepth  <value> - overwrite the range of input image values.\n" \
	" -info - time / performance output is on. Default is off.\n" \
	" -log <file> - enable log file.\n"\
	" -async - enable async mode to overlap processing and file read/write.\n"\
	" -thread <value> - number of processing threads in async mode. Default value is 1.\n"
	" -threadR <value> - number of file reader threads in async mode. Default value is 1.\n"
	" -threadW <value> - number of file writer threads in async mode. Default value is 1.\n"
	"\n" \
	"Command lines for J2K encoder.\n" \
	"Single mode:\n" \
	"J2kEncoderSample.exe -i <input image> -o <output image> -a <name> -c <value> -d <device ID> -info -q <value> -l <value> -s <subsampling> -discard\n" \
	"\n"\
	"Batch mode:\n" \
	"J2kEncoderSample.exe -i <input image> -o <output image> -a <name> -c <value> -d <device ID> -info -q <value> -l <value> -discard -b <value> -repeat <value>\n"\
	"\n"\
	"Multithreaded batch mode:\n" \
	"J2kEncoderSample.exe -i <input image> -o <output image> -a <name> -c <value> -d <device ID> -info -q <value> -l <value> -discard -b <value> -repeat <value> -async -thread 2 -threadR 1 -threadW 1\n";
