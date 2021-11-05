#include "PreloadJpeg.hpp"

fastStatus_t PreloadJpegFromFolder(const char* folderName, fastJfifInfo_t* jfifInfo) {
	std::list<std::string> files;
	{
		getFileList(folderName, files);
		if (files.empty()) {
			fprintf(stderr, "No input files found\n");
			return FAST_IO_ERROR;
		}
	}

	CHECK_FAST(PreloadJpegFromFile(files.begin()->c_str(), jfifInfo));

	files.clear();
	return FAST_OK;
}

fastStatus_t PreloadJpegFromFile(const char* fileName, fastJfifInfo_t* jfifInfo) {
	CHECK_FAST(fastJfifHeaderLoadFromFile(fileName, jfifInfo));
	fastJfifFreeExif(jfifInfo);

	return FAST_OK;
}

fastStatus_t PreloadJpeg(const char* path, bool isFolder, fastJfifInfo_t* jfifInfo) {
	return isFolder ? PreloadJpegFromFolder(path, jfifInfo) : PreloadJpegFromFile(path, jfifInfo);
}
