/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "helper_exif.hpp"

ExifEntry* fastExifInitTag(ExifData *exif, ExifIfd ifd, ExifTag tag) {
	ExifEntry *entry;
	//Return an existing tag if one exists
	if (!((entry = exif_content_get_entry(exif->ifd[ifd], tag)))) {
		// Allocate a new entry
		entry = exif_entry_new();
		if (entry == NULL)
			return 0;
		entry->tag = tag; // tag must be set before calling exif_content_add_entry */

						  // Attach the ExifEntry to an IFD
		exif_content_add_entry(exif->ifd[ifd], entry);

		// Allocate memory for the entry and fill with default data
		exif_entry_initialize(entry, tag);

		// Ownership of the ExifEntry has now been passed to the IFD.
		// One must be very careful in accessing a structure after
		// unref'ing it; in this case, we know "entry" won't be freed
		// because the reference count was bumped when it was added to
		// the IFD.
		exif_entry_unref(entry);
	}
	return entry;
}

ExifEntry* fastExifCreateTag(ExifData *exif, ExifIfd ifd, ExifTag tag, ExifFormat fmt, std::size_t len) {
	ExifEntry *entry = exif_content_get_entry(exif->ifd[ifd], tag);
	if (entry) {
		exif_entry_unref(entry);
	}
	// Create a memory allocator to manage this ExifEntry
	ExifMem *mem = exif_mem_new_default();
	if (mem == NULL)
		return nullptr;

	// Create a new ExifEntry using our allocator
	entry = exif_entry_new_mem(mem);
	if (entry == NULL)
		return nullptr;

	// Allocate memory to use for holding the tag data
	void *buf = exif_mem_alloc(mem, static_cast<unsigned int>(len));
	if (buf == NULL)
		return nullptr;

	// Fill in the entry
	entry->data = (unsigned char*)buf;
	entry->size = static_cast<unsigned int>(len);
	entry->tag = tag;
	entry->components = static_cast<unsigned int>(len);
	entry->format = fmt;

	// Attach the ExifEntry to an IFD
	exif_content_add_entry(exif->ifd[ifd], entry);

	// The ExifMem and ExifEntry are now owned elsewhere
	exif_mem_unref(mem);
	exif_entry_unref(entry);

	return entry;
}

ExifEntry* fastExifCreateAsciiTag(ExifData *exif, ExifIfd ifd, ExifTag tag, unsigned int strlen) {
	ExifEntry* entry = exif_content_get_entry(exif->ifd[ifd], tag);
	if (entry) {
		exif_entry_unref(entry);
	}

	// Create a memory allocator to manage this ExifEntry
	ExifMem *mem = exif_mem_new_default();
	if (mem == NULL)
		return nullptr;

	// Create a new ExifEntry using our allocator
	entry = exif_entry_new_mem(mem);
	if (entry == NULL)
		return nullptr;

	// Allocate memory to use for holding the tag data
	const unsigned int size = exif_format_get_size(EXIF_FORMAT_ASCII) * strlen;
	void *buf = exif_mem_alloc(mem, size);
	if (buf == NULL)
		return nullptr;

	// Fill in the entry
	entry->components = strlen;
	entry->format = EXIF_FORMAT_ASCII;
	entry->tag = tag;
	entry->size = size;
	entry->data = (unsigned char*)buf;

	// Attach the ExifEntry to an IFD
	exif_content_add_entry(exif->ifd[ifd], entry);

	// The ExifMem and ExifEntry are now owned elsewhere
	exif_mem_unref(mem);
	exif_entry_unref(entry);

	return entry;
}

ExifEntry* fastExifCreateUndefinedTag(ExifData *exif, ExifIfd ifd, ExifTag tag, unsigned int len) {
	ExifEntry* entry = exif_content_get_entry(exif->ifd[ifd], tag);
	if (entry) {
		exif_entry_unref(entry);
	}

	// Create a memory allocator to manage this ExifEntry
	ExifMem *mem = exif_mem_new_default();
	if (mem == NULL)
		return nullptr;

	// Create a new ExifEntry using our allocator
	entry = exif_entry_new_mem(mem);
	if (entry == NULL)
		return nullptr;

	// Allocate memory to use for holding the tag data
	const unsigned int size = exif_format_get_size(EXIF_FORMAT_UNDEFINED) * len;
	void *buf = exif_mem_alloc(mem, size);
	if (buf == NULL)
		return nullptr;

	// Fill in the entry
	entry->components = len;
	entry->format = EXIF_FORMAT_UNDEFINED;
	entry->tag = tag;
	entry->size = size;
	entry->data = (unsigned char*)buf;

	// Attach the ExifEntry to an IFD
	exif_content_add_entry(exif->ifd[ifd], entry);

	// The ExifMem and ExifEntry are now owned elsewhere
	exif_mem_unref(mem);
	exif_entry_unref(entry);

	return entry;
}