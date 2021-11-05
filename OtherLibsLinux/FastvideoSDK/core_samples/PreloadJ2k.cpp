#include "PreloadJ2k.hpp"
#include "MallocAllocator.h"

fastStatus_t PreloadJ2kFromFolder(const char* folderName, fastJ2kImageInfo_t* j2kInfo) {
	std::list<std::string> files;
	{
		getFileList(folderName, files);
		if (files.empty()) {
			fprintf(stderr, "No input files found\n");
			return FAST_IO_ERROR;
		}
	}
		
	Bytestream<MallocAllocator> bytestream;
	CHECK_FAST(fvLoadBytestream(*files.begin(), bytestream, false));
	 
	CHECK_FAST(fastDecoderJ2kPredecode(
		j2kInfo,
		bytestream.data.get(),
		bytestream.size
	));

	files.clear();
	return FAST_OK;
}



fastStatus_t PreloadJ2kFromFile(const char* fileName, fastJ2kImageInfo_t* j2kInfo)
{
	Bytestream<MallocAllocator> bytestream;
	CHECK_FAST(fvLoadBytestream(fileName, bytestream, false));

	CHECK_FAST(fastDecoderJ2kPredecode(
		j2kInfo,
		bytestream.data.get(),
		bytestream.size
	));
	return FAST_OK;
}

fastStatus_t PreloadJ2k(const char* path, bool isFolder, fastJ2kImageInfo_t* j2kInfo)
{
	return isFolder ? PreloadJ2kFromFolder(path, j2kInfo) : PreloadJ2kFromFile(path, j2kInfo);
}