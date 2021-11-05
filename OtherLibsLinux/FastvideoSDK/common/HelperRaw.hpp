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

#include "fastvideo_sdk.h"
#include "alignment.hpp"

inline unsigned long GetHostRawPitch(fastRawFormat_t rawFmt, unsigned width) {
	switch (rawFmt)
	{
		case FAST_RAW_XIMEA12: 
			return uSnapUp((unsigned)width, (unsigned)2) * 1.5f;
		case FAST_RAW_PTG12:
			return uSnapUp((unsigned)width, (unsigned)2) * 1.5f;
	};
}

inline unsigned long GetDeviceRawPitch(fastRawFormat_t rawFmt, unsigned width) {
	switch (rawFmt)
	{
		case FAST_RAW_XIMEA12:
			return uSnapUp((unsigned)width, (unsigned)8) * 1.5f;
		case FAST_RAW_PTG12:
			return uSnapUp((unsigned)width, (unsigned)8) * 1.5f;
	};
}
