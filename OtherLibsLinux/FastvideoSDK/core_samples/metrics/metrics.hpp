/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __METRICS_HPP__
#define __METRICS_HPP__

#include <vector>

class Metrics {
private:
	unsigned char *etalonImg;
	unsigned char *testImg;
	unsigned width;
	unsigned height;
	unsigned channels;

	void receiveColor(double *out1, double *out2, int colorOffset);
	
public:
	struct MSEResult {
		double R;
		double G;
		double B;
	};

	Metrics(unsigned char *etalonImg, unsigned char *testImg, unsigned width, unsigned height, unsigned channels) :
			etalonImg(etalonImg), testImg(testImg), width(width), height(height), channels(channels)
	{ 	};

	MSEResult MseByComponents (int skipBorderWidth);
	double MSE (int skipBorderWidth);
	double PSNR (int skipBorderWidth);
	double SSIM(unsigned long offset, unsigned long interleave);
	void DiffGistogram (std::vector<int> &res, int skipBorderWidth);
	void GetPixelDiff(unsigned char *diff, int skipBorderWidth);
};


#endif
