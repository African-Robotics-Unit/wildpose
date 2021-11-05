#include "MultiThreadInfoPrinter.h"
#include "PluralSuffix.h"

void PrintProcessorResults(IAsyncProcessor* processor, BaseOptions* options, const char* pipelineName) {
	AsyncProcessorThreadParam* params = processor->GetResults();

	double requestedMemSize = 0., totalTimeMax = 0., totalTimeAvg = 0.,
		pipelineTimeMax = 0., pipelineTimeAvg = 0., 
		writerTime = 0., readerTime = 0., allocationTime = 0., componentTimeMax = 0.;

	unsigned imageCount = 0;
	for (int i = 0; i < options->NumberOfThreads; i++) {
		requestedMemSize += params[i].result.requestedMemSize;
		totalTimeMax = std::max(totalTimeMax, params[i].result.totalTime);
		totalTimeAvg += params[i].result.totalTime;
		pipelineTimeMax = std::max(pipelineTimeMax, params[i].result.pipelineHostTime);
		pipelineTimeAvg += params[i].result.pipelineHostTime;
		writerTime += params[i].result.writerWaitTime;
		readerTime += params[i].result.readerWaitTime;
		allocationTime += params[i].result.allocationTime;
		imageCount += params[i].result.processedItem;
		componentTimeMax = std::max(componentTimeMax, params[i].result.componentTime);
	}

	totalTimeAvg = totalTimeAvg / double(options->NumberOfThreads);
	pipelineTimeAvg = pipelineTimeAvg / double(options->NumberOfThreads);
	writerTime = writerTime / double(options->NumberOfThreads);
	readerTime = readerTime / double(options->NumberOfThreads);
	allocationTime = allocationTime / double(options->NumberOfThreads);


	printf("Requested GPU memory size: %.2lf MB\n", requestedMemSize);

	if (options->BenchmarkInfo)
	{
		printf("Process time for %d images without HDD I/O and excluding device-to-host transfer = %.2f ms; %.0f FPS\n", imageCount, componentTimeMax * 1000.0, imageCount / (componentTimeMax));
	}
	else if (options->Info) {
		printf("Processed %d image%s on %d processor thread%s.\n", imageCount, PluralSuffix(imageCount), options->NumberOfThreads, PluralSuffix(options->NumberOfThreads));
		printf("Max/Avg Total Time per thread %.1f/%.1f ms; %.1f/%.1f FPS\n", totalTimeMax * 1000.0f, totalTimeAvg * 1000.0f, imageCount / totalTimeMax, imageCount / totalTimeAvg);
		printf("Max/Avg Pipleline Time per thread %.1f/%.1f ms; %.1f/%.1f FPS\n", pipelineTimeMax * 1000.0f, pipelineTimeAvg * 1000.0f, imageCount / pipelineTimeMax, imageCount / pipelineTimeAvg);
		printf("Avg Reader wait time per thread %.1f ms for %d reader thread%s\n", readerTime * 1000.0f, options->NumberOfReaderThreads, PluralSuffix(options->NumberOfReaderThreads));
		printf("Avg Writer wait time per thread %.1f ms for %d writer thread%s\n", writerTime * 1000.0f, options->NumberOfWriterThreads, PluralSuffix(options->NumberOfWriterThreads));
		printf("Avg Allocation time per thread %.1f ms\n", allocationTime * 1000.0f);
	}
	else
		printf("Total %s time including all transfers for %d image%s per %d thread%s = %.1f ms; %.1f FPS;\n",
			pipelineName,
			imageCount, PluralSuffix(imageCount),
			options->NumberOfThreads, PluralSuffix(options->NumberOfThreads),
			pipelineTimeMax * 1000.0f,
			imageCount / pipelineTimeMax
		);
}