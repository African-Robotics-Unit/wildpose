/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
const char *projectName = "ComponentsSample project";
const char *helpProject =
"The ComponentsSample shows how to utilize various SDK modules.\n"
"\n" \
"List of demonstrated components:\n"\
"\tAffine Transformation\n" \
"\tBase Color Correction filter\n" \
"\tBayer Black Shift filter\n" \
"\tSurface Converter\n" \
"\t1D LUTs\n" \
"\tBayer 1D LUTs\n" \
"\tRGB 1D LUTs\n" \
"\t3D RGB LUT\n" \
"\t3D HSV LUT\n" \
"\tSAM and SAM16 filters\n" \
"\tMedian filters\n" \
"\tTone Curve filters\n" \
"\tDefringe filters\n" \
"\tBad Pixel Correction filters\n" \
"\tBinning filters\n" \
"\tGaussian 3x3 filters\n" \
"\n" \
"Supported parameters\n" \
" -i <file>  - input file in BMP/PGM/PPM/TIFF format.\n" \
" -if <folder + mask> - folder path and mask for input file in BMP/PGM/PPM/TIFF formats.\n" \
" -maxWidth <unsigned int> - set maximum image width for multiple file processing.\n" \
" -maxHeight <unsigned int> - set maximum image height for multiple file processing.\n" \
" -o <file> - output file in PGM or PPM format\n" \
" -d <device ID> - GPU device ID. Default is 0.\n" \
" -info - time / performance output is on. Default is off.\n" \
" -log <file> - enable log file.\n"\
"\n" \
" -baseColorCorrection -colorCorrection {matrix} -whiteLevel {R,G,B} - apply base color correction.\n" \
"\tWhere:\n"\
"\t\t -colorCorrection {matrix} - color correction matrix 4x3.\n" \
"\t\t -whiteLevel {R,G,B} - white level.\n" \
" -lut8c, lut12_12c, lut12_8c, lut12_16c, lut16_16c, lut16_8c, lut16_16c_fr - RGB 1D LUTs.\n" \
" -lut8, lut12_12, lut12_8, lut12_16, lut16_16, lut16_8, lut16_16_fr - 1D LUTs.\n" \
" -lut10_16b, -lut12_16b, lut14_16b, lut16_16b - Bayer 1D LUTs.\n"\
" -lut_r <LUT file>, lut_g <LUT file>, lut_b <LUT file> - (for Bayer and RGB 1D LUTs) LUT files for R, G, B channels respectively.\n"\
" -lut <LUT file> - (for 1D LUTs) LUT file. Simple txt file.\n"\
" -pattern <pattern> - (for Bayer 1D LUTs and -bayerBlackShift) bayer pattern (RGGB, BGGR, GBRG, GRBG).\n"\
" -affine <affine transform> - affine transform (can be flip, flop, rotate180, rotate90left, rotate90right).\n"\
" -toneCurve <file> - apply tonecurve filter from selected file \n"\
" -sam, -sam16 -matrixA <file> -matrixB <file> - apply sam filter.  \n"\
"\tWhere:\n"\
"\t\t -matrixA <file> - (for -sam and -sam16) file with intensity correction values for each image pixel. File is in PFM format.\n"\
"\t\t -matrixB <file> - (for -sam and -sam16) file with black shift values for each image pixel. File is in PGM format. For SAM16 contains 16 bit values.\n"\
"\t\t -median - apply median filter.\n"\
" -bitDepthConverter -dstBitsPerChannel <int> - change image bit depth.\n"\
"\tWhere:\n"\
"\t\t -dstBitsPerChannel <int> - (for -bitDepthConverter) new image bit depth.\n"\
" -bayerBlackShift -blackShiftR <int> -blackShiftG <int> -blackShiftB <int> - apply Bayer Black Shift filter.\n"\
"\tWhere:\n"\
"\t\t -blackShiftR <int>, -blackShiftG <int>, -blackShiftB <int> - shift values for R, G, B channels respectively.\n"\
" -colorConvertion - convert between color and gray images.\n"\
" -bgrxImport - conver image from BGRX8 format to RGB.\n"\
" -selectChannel -channel <type> - convert color image to gray by selecting one channel.\n"\
"\tWhere:\n"\
"\t\t -channel <type> - (for -selectChannel) channel are R or G or B.\n"\
" -rgbLut3D -lut <file> - apply RGB 3D LUT.\n"\
"\tWhere:\n"\
"\t\t -lut <file> - (for -rgbLut3D) file format is .cube.\n"\
" -hsvLut3D -lut <file> - apply HSV 3D LUT.\n"\
"\tWhere:\n"\
"\t\t -lut <file> - (for -hsvLut3D) file forma is .xml.\n"\
" -defringe -window <int> -tintR <int> -tintG <int> -tintB <int> -coefficient <float> -fi_max <int> - apply Defringe filter. Only color image is supported.\n"\
"\tWhere:\n"\
"\t\t -window <int> - set window size. Maximum size 40.\n"\
"\t\t -tintR <int>, -tintG <int>, -tintB <int>- defines center of unwanted colors region. Range is defined by image bit depth.\n"\
"\t\t -fi_max <int> - defines angle (in degree)  in CbCr plane for unwanted colors region.\n"\
"\t\t -coefficient <float> - relative luma threshold. Pixels with unwanted colors below threshold are ignored.\n"\
" -badPixelCorrection -pattern <pattern> - apply Bad Pixel Correction filter. Grayscale and bayer images are supported.\n"\
"\tWhere:\n"\
"\t\t -pattern <pattern> - define bayer pattern {RGGB, BGGR, GBRG, GRBG} for bayer images.\n"\
" -binning -mode {avg, sum} -factor {2, 3, 4} - apply binning filter. Only grayscale images are supported.\n"\
"\tWhere:\n"\
"\t\t -mode {avg, sum} - binning mode.\n"\
"\t\t -factor {2, 3, 4} - binning factor.\n"\
" -sharp -sigma {float} - apply fast Gaussian Sharpen 3x3 filter. Only 8-bits images are supported.\n"\
"\tWhere:\n"\
"\t\t -sigma {float} - filter parameter.\n"\
" -blur -sigma {float} - apply fast Gaussian Blur 3x3 filter. Only 8-bits images are supported.\n"\
"\tWhere:\n"\
"\t\t -sigma {float} - filter parameter.\n"\
"\n" \
"Example of command line for base color correction:\n" \
"ComponentsSample.exe -baseColorCorrection -i input.ppm -o output.ppm  -colorCorrection {1,0,0,0,0,1,0,0,0,0,1,0} -whiteLevel {255,255,255} -info\n"\
"\n" \
"Example of command line for 1D LUT:\n" \
"ComponentsSample.exe -lut8 -i input.ppm  -o output.ppm -lut lut.txt -info\n" \
"\n" \
"Example of command line for RGB 1D LUT:\n" \
"ComponentsSample.exe -lut8c -i input.ppm  -o output.ppm -lut_r lutR.txt -lut_g lutG.txt -lut_b lutB.txt -info\n" \
"\n" \
"Example of command line for Bayer 1D LUT:\n" \
"ComponentsSample.exe -lut12_16b -i input.pgm  -o output.pgm -lut_r lutR.txt -lut_g lutG.txt -lut_b lutB.txt -pattern RGGB -info\n" \
"\n" \
"Example of command line for affine transform:\n" \
"ComponentsSample.exe -affine flip -i input.ppm -o output.ppm -info\n"\
"\n" \
"Example of command line for tone curve::\n" \
"ComponentsSample.exe -toneCurve toneCurve.txt -i input.ppm -o output.ppm  -info\n"\
"\n" \
"Example of command line for SAM:\n" \
"ComponentsSample.exe -sam -i input.ppm -o output.ppm -matrixA matrixA.pfm -matrixB matrixB.pgm -info\n"\
"\n" \
"Example of command line for median:\n" \
"ComponentsSample.exe -median -i input.ppm -o output.ppm -info\n"\
"\n" \
"Example of command line to change image bit depth:\n" \
"ComponentsSample.exe -bitDepthConverter -i input.ppm -o output.ppm -dstBitsPerChannel 16 -info\n"\
"\n" \
"Example of command line for Bayer Black Shift:\n" \
"ComponentsSample.exe -bayerBlackShift -i input.pgm  -o output.pgm -blackShiftR 64 -blackShiftG 64 -blackShiftB 64 -pattern RGGB -info\n" \
"\n" \
"Example of command line to convert color image to gray:\n" \
"ComponentsSample.exe -rgbToGray -i input.ppm -o output.pgm -info\n"\
"ComponentsSample.exe -selectChannel -i input.ppm -o output.pgm  -channel R -info\n"\
"\n" \
"Example of command line for RGB 3D LUT:\n" \
"ComponentsSample.exe -rgbLut3D -i input.ppm  -o output.ppm -lut lut.cube -info\n"\
"\n" \
"Example of command line for HSV 3D LUT:\n" \
"ComponentsSample.exe -hsvLut3D -i input.ppm -o output.ppm -lut lut.xml -info\n"\
"\n" \
"Example of command line for Defringe:\n" \
"ComponentsSample.exe -defringe -i input.ppm -o output.ppm -window 40 -tintR 110 -tintG 113 -tintB 108 -coefficient 0.05 -fi_max 60  -info\n"\
"\n" \
"Example of command line for Binning:\n" \
"ComponentsSample.exe -binning -i input.pgm -o output.pgm -mode avg -factor 2 -info\n"\
"\n" \
"Example of command line for Bad Pixel Correction:\n" \
"ComponentsSample.exe -badPixelCorrection -i input.pgm -o output.pgm -pattern RGGB -info\n";



