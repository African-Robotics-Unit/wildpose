/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __HELPER_EXIF__
#define __HELPER_EXIF__

#include <cstddef>

#include "libexif/exif-data.h"

#define EXIF_SECTION_CODE 0xFFE1

ExifEntry* fastExifInitTag(ExifData *exif, ExifIfd ifd, ExifTag tag);
ExifEntry* fastExifCreateTag(ExifData *exif, ExifIfd ifd, ExifTag tag, ExifFormat fmt, std::size_t len);
ExifEntry* fastExifCreateAsciiTag(ExifData *exif, ExifIfd ifd, ExifTag tag, unsigned int strlen);
ExifEntry* fastExifCreateUndefinedTag(ExifData *exif, ExifIfd ifd, ExifTag tag, unsigned int len);

#endif // __HELPER_EXIF__