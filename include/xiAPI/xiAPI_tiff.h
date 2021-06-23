
#pragma once

// -------------------------------------------------------------
// Write image to file
// If error - throw char* description
void WriteImage(XI_IMG* image, char* filename);
void xiImageGetBitCount(XI_IMG* image, int* count);
