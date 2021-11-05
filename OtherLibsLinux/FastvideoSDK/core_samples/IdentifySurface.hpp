/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __IDENTIFY_SURFACE__
#define __IDENTIFY_SURFACE__

static fastSurfaceFormat_t IdentifySurface(const unsigned bitsPerChannel, const unsigned nChannels) {
	if	(bitsPerChannel<= 8)
			return nChannels == 1 ? FAST_I8 : FAST_RGB8;
	if (bitsPerChannel <= 10 && nChannels == 1)
			return FAST_I10;
	if (bitsPerChannel <= 12)
			return nChannels == 1 ? FAST_I12 : FAST_RGB12;
	if (bitsPerChannel <= 14 && nChannels == 1)
			return FAST_I14;
	if (bitsPerChannel <= 16)
			return nChannels == 1 ? FAST_I16 : FAST_RGB16;

	return FAST_I8;
}

#endif //__SURFACE_TRAITS__