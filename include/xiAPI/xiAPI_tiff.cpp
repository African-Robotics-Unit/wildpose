// this library is able to save xiAPI image into TIFF file

#include <stdio.h>
#include <memory.h>
#ifdef WIN32
#include <xiApi.h>       // Windows
#include <libtiff/tiffio.h>
#else
#include <m3api/xiApi.h> // Linux, OSX
#include <tiffio.h>
#endif

void xiImageGetBitCount(XI_IMG* image, int* count)
{
	int bit_count = 1;
	switch (image->frm)
	{
	case XI_MONO8:
	case XI_RAW8:
	case XI_RAW8X2:
	case XI_RAW8X4:
	case XI_RGB_PLANAR:	bit_count = 8; break;
	case XI_MONO16:
	case XI_RAW16:
	case XI_RAW16X2:
	case XI_RAW16X4:
	case XI_RGB16_PLANAR: bit_count = 16; break;
	case XI_RGB24:		bit_count = 24; break;
	case XI_RAW32:
	case XI_RGB32:
	case XI_RAW32FLOAT:	bit_count = 32; break;
	case XI_RGB48:		bit_count = 48; break;
	case XI_RGB64:		bit_count = 64; break;
	default:
		throw("xiImageGetBitCount - Unsupported data format\n");
	}
	*count = bit_count;
}

void WriteImage(XI_IMG* image, char* filename)
{
	TIFF* tiff_img = TIFFOpen(filename, "w");
	if (!tiff_img)
		throw "Opening image by TIFFOpen";

	// set tiff tags
	int width = image->width;
	int height = image->height;
	int bits_per_sample = 1;
	xiImageGetBitCount(image, &bits_per_sample);
	int line_len = 0;
	line_len = width * (bits_per_sample / 8);
	bits_per_sample = 8;
	printf("Saving image %dx%d to file:%s\n", width, height, filename);
	printf("Bps %d max sample %d\n", bits_per_sample, ((1 << bits_per_sample) - 1));

	TIFFSetField(tiff_img, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(tiff_img, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(tiff_img, TIFFTAG_SAMPLESPERPIXEL, 4);
	TIFFSetField(tiff_img, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
	TIFFSetField(tiff_img, TIFFTAG_MINSAMPLEVALUE, 0);
	TIFFSetField(tiff_img, TIFFTAG_MAXSAMPLEVALUE, (1 << bits_per_sample) - 1);
	TIFFSetField(tiff_img, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(tiff_img, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(tiff_img, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
	TIFFSetField(tiff_img, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
	TIFFSetField(tiff_img, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);
	TIFFSetField(tiff_img, TIFFTAG_ROWSPERSTRIP, height);

	// save data
	if (TIFFWriteEncodedStrip(tiff_img, 0, image->bp, line_len*height) == -1)
	{
		throw("ImageFailed to write image");
	}

	TIFFWriteDirectory(tiff_img);
	TIFFClose(tiff_img);
}
