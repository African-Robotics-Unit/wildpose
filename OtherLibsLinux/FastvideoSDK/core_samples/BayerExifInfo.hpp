/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __EXIF_INFO__
#define __EXIF_INFO__

#include "fastvideo_sdk.h"
#include "helper_exif.hpp"

#define EXIF_NAME "splitted bayer\0"
#define EXIF_NAME_LENGTH 15

#pragma pack(1)
typedef struct {
	char name[EXIF_NAME_LENGTH];
	char bayerFmt;
	unsigned width;
	unsigned height;
} fastSplitterExif_t;
#pragma pack()

static fastJpegExifSection_t *GenerateSplitterExif( fastBayerPattern_t bayerFmt, unsigned width, unsigned height ) {

	unsigned char* exifBytestream;
	unsigned int   exifBytestreamLen;

	ExifEntry* entry;
	ExifData* exifData = exif_data_new();
	
	exif_data_unset_option(exifData, EXIF_DATA_OPTION_FOLLOW_SPECIFICATION);
	exif_data_set_option(exifData,EXIF_DATA_OPTION_DONT_CHANGE_MAKER_NOTE);
	exif_data_set_byte_order(exifData, EXIF_BYTE_ORDER_MOTOROLA);

	fastSplitterExif_t splitterInfo;

	strcpy(splitterInfo.name, EXIF_NAME);
	splitterInfo.bayerFmt = (char)bayerFmt;
	splitterInfo.width = width;
	splitterInfo.height = height;

	entry = fastExifCreateUndefinedTag(exifData, EXIF_IFD_EXIF, EXIF_TAG_MAKER_NOTE, sizeof(fastSplitterExif_t));
	memcpy((char *)entry->data, (char*)&splitterInfo, sizeof(fastSplitterExif_t));
	exif_data_save_data(exifData, &exifBytestream, &exifBytestreamLen);

	fastJpegExifSection_t *exifSection = new fastJpegExifSection_t[1];
	exifSection->exifCode = EXIF_SECTION_CODE;
	exifSection->exifLength = exifBytestreamLen;
	exifSection->exifData = new char[exifBytestreamLen];
	memcpy(exifSection->exifData, exifBytestream, exifBytestreamLen);

	if (exifBytestream)
		free(exifBytestream);

	exif_data_free(exifData);
	return exifSection;
}

static fastStatus_t UpdateSplitterExif( fastJpegExifSection_t *exifSection, unsigned width, unsigned height ) {


	ExifData* exifData = exif_data_new();

	exif_data_unset_option(exifData, EXIF_DATA_OPTION_FOLLOW_SPECIFICATION);
	exif_data_set_option(exifData, EXIF_DATA_OPTION_DONT_CHANGE_MAKER_NOTE);
	exif_data_load_data(exifData, (unsigned char*)exifSection->exifData, exifSection->exifLength);
	ExifEntry* entry;

	unsigned char* exifBytestream;
	unsigned int   exifBytestreamLen;


	entry = exif_content_get_entry(exifData->ifd[EXIF_IFD_EXIF], EXIF_TAG_MAKER_NOTE);
	if (entry) {
		fastSplitterExif_t* splitterInfo = (fastSplitterExif_t*)entry->data;
		if (strcmp(splitterInfo->name, EXIF_NAME)==0)
		{
			splitterInfo->width = width;
			splitterInfo->height = height;

			exif_data_save_data(exifData, &exifBytestream, &exifBytestreamLen);
			memcpy(exifSection->exifData, exifBytestream, exifBytestreamLen);

			if (exifBytestream)
				free(exifBytestream);
			exif_data_free(exifData);
			return FAST_OK;
		}
	}
	exif_data_free(exifData);
	return FAST_INVALID_VALUE;
}

static fastStatus_t ParseSplitterExif( fastJpegExifSection_t *exifSection, fastBayerPattern_t &pattern, unsigned &width, unsigned &height ) {

	ExifData* exifData = exif_data_new();
	
	exif_data_unset_option(exifData, EXIF_DATA_OPTION_FOLLOW_SPECIFICATION);
	exif_data_set_option(exifData, EXIF_DATA_OPTION_DONT_CHANGE_MAKER_NOTE);
	exif_data_load_data(exifData, (unsigned char*)exifSection->exifData, exifSection->exifLength);

	ExifEntry* entry;

	entry = exif_content_get_entry(exifData->ifd[EXIF_IFD_EXIF], EXIF_TAG_MAKER_NOTE);
	if (entry) {
		fastSplitterExif_t* splitterInfo = (fastSplitterExif_t*)entry->data;
		if (strcmp(splitterInfo->name, EXIF_NAME)==0)
		{
			pattern = (fastBayerPattern_t)splitterInfo->bayerFmt;
			width = splitterInfo->width;
			height = splitterInfo->height;
			exif_data_free(exifData);
			return FAST_OK;
		}
	}
	exif_data_free(exifData);
	return FAST_INVALID_VALUE;
}

#endif // __EXIF_INFO__