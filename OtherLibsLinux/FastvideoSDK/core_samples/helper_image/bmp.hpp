/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __BMP_HPP__
#define __BMP_HPP__

#include "BaseAllocator.h"

////////////////////////////////////////////////////////////////////////////////
// BMP load / store
////////////////////////////////////////////////////////////////////////////////

typedef enum{
	BMP_OK,                           //No error
	BMP_IO_ERROR,                     //Failed to open/access file
	BMP_INVALID_FORMAT,               //Invalid file format
	BMP_UNSUPPORTED_FORMAT          //File format is unsupported by the current version of FAST
} bmpStatus_t;

int LoadHeaderBMP(
    const char* fname,
    unsigned& width,
    unsigned& height,
    unsigned& numberOfChannels,
    unsigned& bitsPerChannel
);

bmpStatus_t LoadBMP(
	void**				data,
	BaseAllocator		*alloc,
    unsigned            &surfaceHeight,
    unsigned            &surfaceWidth,
    unsigned            &surfacePitch8,
	unsigned			&channels,
    const char *filename
);

bmpStatus_t StoreBMP(
    const char           *filename,
	unsigned char 		 *h_Surface,
	unsigned			  channels,
    unsigned              surfaceHeight,
    unsigned              surfaceWidth,
    unsigned              surfacePitch8
);

#endif
