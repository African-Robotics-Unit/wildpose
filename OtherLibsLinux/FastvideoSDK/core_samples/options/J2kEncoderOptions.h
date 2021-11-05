/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __J2K_ENCODER_OPTIONS__
#define __J2K_ENCODER_OPTIONS__

#include "BaseOptions.h"

#include "fastvideo_encoder_j2k.h"

class J2kEncoderOptions : public BaseOptions {
public:
	char *AlgorithmName;
	fastEncoderJ2kAlgorithmType_t Algorithm;
    long InputFilesize;
    long TargetFilesize;
    float CompressionRatio;
    float Quality;
	int CodeblockSize;
	int DWT_Levels;
	int Tier2Threads;
	bool Discard;
	bool NoHeader;
    bool NoMCT;
	int TileWidth;
	int TileHeight;
	int OutputBitDepth;
	int OverwriteSurfaceBitDepth;
	int ss1_x, ss1_y, ss2_x, ss2_y, ss3_x, ss3_y;

    float Timeout;

	J2kEncoderOptions(void) {};
	~J2kEncoderOptions(void) {};

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __J2K_ENCODER_OPTIONS__
