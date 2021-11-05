/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

const char *projectName = "J2KDecoderSample project";
const char *helpProject = "The J2KDecoderSample demonstrates JPEG2000 decoder functionality.\n"\
"\n" \
"There are three modes for J2K decoder sample application : single, batch, multithreaded batch.\n" \
"\n" \
"1. In the single mode every image is decoded one by one. It is the minimum latency mode with minimal memory consumption. Single mode is activated by default.\n" \
"\n" \
"2. In the batch mode, multiple images are decoded in one instance of the decoder simultaneously. Batch mode has better performance than single mode, but consumes much more GPU memory. Parameter -b defines batch size. Batch mode is activated when parameter -b is greater than one.\n" \
"\n" \
"In this sample you can easily test the batch mode starting repeated processing of one image via parameter -repeat. Faster decoding is achieved when -repeat value is multiple of -b.\n" \
"\n" \
"3. In the multithreaded batch mode we run two batch J2K decoders in separate threads. This allows us to overlap computations and data transfers to increase GPU utilization. This mode shows the maximum performance and quite high GPU memory usage. Setting -threads parameter to 2 activates this mode. Also you can increase number of threads over 2 to get even better performance. Use parameter -repeat to supply decoders with enough workload.\n" \
"\n" \
"Supported parameters\n" \
" -i <file> - input file in JP2/J2K format.\n" \
" -if <folder + mask> - folder path and mask for input file in BMP/PGM/PPM/TIFF formats.\n" \
" -o <file> - output file in PGM/PPM format.\n" \
" -maxWidth <value> - set maximum image width for multiple file processing.\n" \
" -maxHeight <value> - set maximum image height for multiple file processing.\n" \
" -dynamicAllocation - enables memory reallocation when the next image requires more memory than was allocated.\n" \
" -t2thread <value> - set the number of CPU threads in single mode Tier-2 stage for tiled images.\n" \
" -sequentialTiles - sequential tile processing when the whole image is too large to fit in memory.\n" \
" -memoryLimit <value> - amount of GPU memory (in megabytes) available to decoder. It only affects tiled images.\n" \
" -d <value> - GPU (device) index.\n" \
" -b <value> - batch size (maximum number of simultaneously processed images).\n" \
" -maxTileWidth - set maximum tile width for multiple file processing.\n" \
" -maxTileHeight - set maximum tile height for multiple file processing.\n" \
" -window <width x height + offsetX + offsetY> - decode selected ROI. It defines ROI size and left corner offset.\n" \
" -repeat <value> - how many times to repeat encoding for one image.\n" \
" -discard - do not write the output to disk.\n" \
" -bits <value> - decode the specified number of bitplanes\n" \
" -passes <value> - decode the specified number of passes. Max value for passes is 1 + (image bitdepth - 1) * 3\n" \
" -forceTo8bits - decoded image convert to 8 bits bitdepth.\n" \
" -printgml - print GML (Geography Markup Language) data for geographic image.\n" \
" -info - time / performance output is on. Default is off.\n" \
" -log <file> - enable log file.\n"\
" -async - enable async mode to overlap processing and file read/write.\n"\
" -thread <value> - number of processing threads in async mode. Default value is 1.\n"
" -threadR <value> - number of file reader threads in async mode. Default value is 1.\n"
" -threadW <value> - number of file writer threads in async mode. Default value is 1.\n"
"\n" \
"Command lines for J2K decoder.\n" \
"Single mode:\n" \
"J2KDecoderSample.exe -i <input image> -o <output image> -d <device ID> -info -discard\n" \
"\n"\
"Batch mode:\n" \
"J2KDecoderSample.exe -i <input image> -o <output image> -d <device ID> -info -discard -b <value> -repeat <value>\n"\
"\n"\
"Multithreaded batch mode:\n" \
"J2KDecoderSample.exe -i <input image> -o <output image> -d <device ID> -info -discard -b <value> -repeat <value> -async -thread 2 -threadR 1 -threadW 1\n";

