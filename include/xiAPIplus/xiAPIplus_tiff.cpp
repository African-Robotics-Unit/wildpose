// this library is able to save xiAPIplus image into TIFF file

#include <xiAPIplus/xiapiplus.h>
#ifdef WIN32
#include <libtiff/tiffio.h>
#else
#include <tiffio.h>
#endif

void WriteImage(xiAPIplus_Image* image, char* filename)
{
	TIFF* tiff_img = TIFFOpen(filename, "w");
	if (!tiff_img)
		throw "Opening image by TIFFOpen";

	// set tiff tags
	int width = image->GetWidth();
	int height = image->GetHeight();
	int bits_per_sample = image->GetBitCount();
	int line_len = 0;
	line_len = image->GetWidth() * (bits_per_sample / 8);
	printf("Saving image %dx%d to file:%s\n", width, height, filename);

	TIFFSetField(tiff_img, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(tiff_img, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(tiff_img, TIFFTAG_SAMPLESPERPIXEL, 1);
	TIFFSetField(tiff_img, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
	TIFFSetField(tiff_img, TIFFTAG_MINSAMPLEVALUE, 0);
	TIFFSetField(tiff_img, TIFFTAG_MAXSAMPLEVALUE, (1 << bits_per_sample) - 1);
	TIFFSetField(tiff_img, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(tiff_img, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	if (XI_RGB24 == image->GetDataFormat())
		TIFFSetField(tiff_img, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
	else
		TIFFSetField(tiff_img, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
	TIFFSetField(tiff_img, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
	TIFFSetField(tiff_img, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);
	TIFFSetField(tiff_img, TIFFTAG_ROWSPERSTRIP, height);

	// save data
	if (TIFFWriteEncodedStrip(tiff_img, 0, image->GetPixels(), line_len*height) == -1)
	{
		throw("ImageFailed to write image");
	}

	TIFFWriteDirectory(tiff_img);
	TIFFClose(tiff_img);
}
