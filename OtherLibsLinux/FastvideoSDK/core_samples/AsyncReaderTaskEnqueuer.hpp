/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#pragma once

#include <thread>
#include "AsyncFileReader.hpp"
#include <list>
#include "supported_files.hpp"

class AsyncReaderTaskEnqueuer {
protected:
	volatile bool* terminateAll;
	IAsyncFileReader* fileReader;
	std::thread* enqueuerThread;

	unsigned repeat;

	virtual void EnqueuerFunc() = 0;
	
	void EnqueuerThreadFunc() {
		EnqueuerFunc();
	}

public:
	AsyncReaderTaskEnqueuer() { 
		enqueuerThread = nullptr;
		fileReader = nullptr;
		terminateAll = nullptr;
	}

	void Init(IAsyncFileReader *fileReader, unsigned repeat, volatile bool* terminateAll) {
		this->terminateAll = terminateAll;
		this->fileReader = fileReader;
		this->repeat = repeat;
		
		enqueuerThread = new std::thread(&AsyncReaderTaskEnqueuer::EnqueuerThreadFunc, this);
	}

	void WaitAll() {
		enqueuerThread->join();
	}

	~AsyncReaderTaskEnqueuer(){
		delete enqueuerThread;
	}
};

class AsyncReaderTaskEnqueuerForDirWithRepeat : public AsyncReaderTaskEnqueuer {
protected:
	const char* dirPath;
	const char* outPath;

	void EnqueuerFunc() {
		std::list<std::string> dirFiles;

		getFileList(dirPath, dirFiles);

		if (dirFiles.empty()) {
			fprintf(stderr, "No input files found\n");
			fileReader->EnqueNextFile(nullptr);
			return;
		}

		int idx = 0;
		for (auto file = dirFiles.begin(); file != dirFiles.end() && !(*terminateAll); file++)
		{
			for (int i = 0; i < repeat && !(*terminateAll); i++)
			{
				File_t fileTask;
				fileTask.inputFileName = *file;
				fileTask.outputFileName = generateOutputFileName(outPath, idx);
				idx++;
				fileReader->EnqueNextFile(&fileTask);
			}
		}
		fileReader->EnqueNextFile(nullptr);
	}
public:
	void Init(const char* dirPath, const char *outPath, unsigned repeat, IAsyncFileReader* fileReader, volatile bool* terminateAll)
	{
		AsyncReaderTaskEnqueuer::Init(fileReader, repeat, terminateAll);
		this->dirPath = dirPath;
		this->outPath = outPath;
		
	}
};

class AsyncReaderTaskEnqueuerForSingleWithRepeat : public AsyncReaderTaskEnqueuer
{
protected:
	const char* file;
	const char* outPath;

	void EnqueuerFunc()
	{
		int idx = 0;
		for (int i = 0; i < repeat && !(*terminateAll); i++)
		{
			File_t fileTask;
			fileTask.inputFileName = file;
			fileTask.outputFileName = generateOutputFileName(outPath, idx);
			idx++;
			fileReader->EnqueNextFile(&fileTask);
		}
		fileReader->EnqueNextFile(nullptr);
	}
public:
	void Init(const char* file, const char* outPath, unsigned repeat, IAsyncFileReader* fileReader, volatile bool* terminateAll)
	{
		AsyncReaderTaskEnqueuer::Init(fileReader, repeat, terminateAll);
		this->file = file;
		this->outPath = outPath;

	}

};
