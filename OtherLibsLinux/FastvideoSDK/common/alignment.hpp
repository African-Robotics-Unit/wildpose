/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

////////////////////////////////////////////////////////////////////////////////
// Common math
////////////////////////////////////////////////////////////////////////////////
template<typename T> static inline T uDivUp(T a, T b){
    return (a / b) + (a % b != 0);
}

template<typename T> static inline T uSnapUp(T a, T b){
    return ( a + (b - a % b) % b );
}

template<typename T> static inline T uSnapDown(T a, T b) {
	return (a - (a % b));
}

template<typename T> static inline T uSnapDelta(T a, T b){
    return (b - a % b) % b;
}

unsigned GetAlignedPitch(unsigned width, unsigned channels , unsigned bytePerChannel, unsigned boundary);
unsigned GetAlignedPitch(unsigned width, unsigned channels, float bytePerChannel, unsigned boundary);

#endif
