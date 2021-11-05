/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "BaseAllocator.h"
#include "IdentifySurface.hpp"
#include "SurfaceTraits.hpp"

#include "tiff.h"
#include "tiffio.h"

#include <cstring>

int LoadHeaderTIFF(
	const char* fname,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel
) {
	TIFF* tif = TIFFOpen(fname, "r");
	if (!tif)
		return 0;

	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

	TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &numberOfChannels);
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerChannel);

	TIFFClose(tif);
}

int LoadTIFF(
	const char* file,
	void** data,
	BaseAllocator* alloc,
	unsigned& width,
	unsigned& pitch,
	unsigned& height,
	unsigned& bitsPerChannel,
	unsigned& channels
) {
	unsigned allocatedSize = 0;

	TIFF* tif = TIFFOpen(file, "r");
	if (!tif) 
		return 0;

	unsigned numberOfDirectories = TIFFNumberOfDirectories(tif);
	if (numberOfDirectories > 1) {
		fprintf(stderr, "Number of directories more than 1 (%d)\n", numberOfDirectories);
		return 0;
	}

	{
		TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

		TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &channels);
		TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerChannel);

		unsigned tileWidth = 0, tileHeight = 0/*, tileImageDepth*/;
		TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
		TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);
		//TIFFGetField(tif, TIFFTAG_IMAGEDEPTH, &tileImageDepth);
		
		unsigned bytesPerChannel = (bitsPerChannel + 7) / 8;
		pitch = GetAlignedPitch(width, channels, bytesPerChannel, alloc->getAlignment());
		*data = (unsigned char *)alloc->allocate(pitch * height);

		unsigned char* img = (unsigned char*)(*data);
		uint32 config = 0;
		TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &config);

		if (config == PLANARCONFIG_CONTIG) {
			if (tileWidth > 0 && tileWidth < width) {
				const unsigned tileRowSize = TIFFTileRowSize(tif);
				const unsigned tileSize = TIFFTileSize(tif);

				tdata_t buf = _TIFFmalloc(tileSize);
				for (uint32 y = 0; y < height; y += tileHeight) {
					for (uint32 x = 0; x < width; x += tileWidth) {
						TIFFReadTile(
							tif,
							buf,
							x, y, 0/*up to tileImageDepth*/,
							0/*used with PLANARCONFIG_SEPARATE*/
						);
						for (uint32 tileY = 0; tileY < tileHeight && (y + tileY) < height; tileY++) {
							memcpy(
								&img[(y + tileY) * pitch + x * channels * bytesPerChannel],
								&((unsigned char *)buf)[tileY * tileRowSize],
								std::min(tileWidth, width - x) * channels * bytesPerChannel
							);
						}
					}
				}
				_TIFFfree(buf);
			} else {
				tdata_t buf = _TIFFmalloc(TIFFScanlineSize(tif));
				for (uint32 row = 0; row < height; row++) {
					TIFFReadScanline(tif, buf, row);
					memcpy(&img[row * pitch], buf, width * channels * bytesPerChannel);

				}
				_TIFFfree(buf);
			}
		} /*else if (config == PLANARCONFIG_SEPARATE) {
			for (uint16 s = 0; s < nsamples; s++)
				for (uint32 row = 0; row < height; row++)
					TIFFReadScanline(tif, buf, row, s);
		}*/
	}

	TIFFClose(tif);
	return 1;
}

/******************************************************************/
/******************************************************************/
/*https://gist.github.com/Trass3r/8866e6d5b5a2c808135268b5bedeccea*/
/******************************************************************/
/******************************************************************/

enum class TYPE : uint16_t {
	None = 0,
	BYTE, ASCII, SHORT, LONG, RATIONAL,
	SBYTE, UNDEF, SSHORT, SLONG, SRATIONAL,
	FLOAT, DOUBLE
};

enum class TAG : uint16_t {
	None = 0,
	NewSubFileType = 254, SubFileType, ImageWidth, ImageLength, BitsPerSample, Compression,
	PhotometricInterpretation = 262,
	Make = 271, Model, StripOffsets, Orientation,
	SamplesPerPixel = 277, RowsPerStrip, StripByteCounts, MinSampleValue, MaxSampleValue, XResolution, YResolution, PlanarConfiguration
};

struct TiffTag {
	TAG      tag;
	TYPE     type;
	uint32_t count;
	uint32_t data;
};

int SaveTIFF(
	const char* filename,
	const void* _data,
	uint32_t width,
	uint32_t pitch,
	uint32_t height,
	uint32_t elsize,
	uint32_t channels
) {
	const uint8_t* data = (uint8_t*)_data;
	FILE* fptr = fopen(filename, "wb");
	std::unique_ptr<FILE, int(*)(FILE*)> closer(fptr, &fclose);

	// write little endian header and directory offset
	uint32_t val = 0x002a4949;
	fwrite(&val, sizeof(val), 1, fptr);
	val = width * height * channels * elsize + 8;
	fwrite(&val, sizeof(val), 1, fptr);

	// write data
	const unsigned lineSizeInByte = width * channels * elsize;
	for (int i = 0; i < height; i++) {
		fwrite(&data[pitch * i], 1, lineSizeInByte, fptr);
	}

	// write IFD
	uint16_t numEntries = 8;
	fwrite(&numEntries, sizeof(numEntries), 1, fptr);

	TiffTag tag;

	tag = { TAG::ImageWidth, TYPE::LONG, 1, width };
	fwrite(&tag, sizeof(tag), 1, fptr);

	tag = { TAG::ImageLength, TYPE::LONG, 1, height };
	fwrite(&tag, sizeof(tag), 1, fptr);

	tag = { TAG::BitsPerSample, TYPE::SHORT, 1, elsize * 8 };
	fwrite(&tag, sizeof(tag), 1, fptr);

	tag = { TAG::PhotometricInterpretation, TYPE::SHORT, 1, channels == 3 ? 2u : 32803 };
	fwrite(&tag, sizeof(tag), 1, fptr);

	tag = { TAG::StripOffsets, TYPE::LONG, 1, 8 }; // data comes right after file header
	fwrite(&tag, sizeof(tag), 1, fptr);

	tag = { TAG::SamplesPerPixel, TYPE::SHORT, 1, channels };
	fwrite(&tag, sizeof(tag), 1, fptr);

	tag = { TAG::RowsPerStrip, TYPE::LONG, 1, width };
	fwrite(&tag, sizeof(tag), 1, fptr);

	tag = { TAG::StripByteCounts, TYPE::LONG, 1, width * height * channels * elsize };
	fwrite(&tag, sizeof(tag), 1, fptr);

	uint32_t offsetNextIFD = 0;
	fwrite(&offsetNextIFD, sizeof(offsetNextIFD), 1, fptr);

	return 1;
}
