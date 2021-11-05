/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include "metrics.hpp"
#include "SSIM.h"

Metrics::MSEResult Metrics::MseByComponents (int skipBorderWidth)
{
	MSEResult res;

	int w3 = width*channels;
	double eG = 0.0, eB = 0.0, eR = 0.0;
	double wh = width*height;

	for (int i=skipBorderWidth; i<(int)height-skipBorderWidth;i++)
	  for (int j=skipBorderWidth; j<(int)width-skipBorderWidth;j++)
		{
			eR += (etalonImg[w3 * i + j * channels    ] - testImg[w3 * i + j * channels    ]) * (etalonImg[w3 *i + j * channels    ] - testImg[w3 * i + j * channels    ]);
			eG += (etalonImg[w3 * i + j * channels + 1] - testImg[w3 * i + j * channels + 1]) * (etalonImg[w3 *i + j * channels + 1] - testImg[w3 * i + j * channels + 1]);
			eB += (etalonImg[w3 * i + j * channels + 2] - testImg[w3 * i + j * channels + 2]) * (etalonImg[w3 *i + j * channels + 2] - testImg[w3 * i + j * channels + 2]);
		}
	
	res.R = eR/wh;
	res.G = eG/wh;
	res.B = eB/wh;
	return res;
}


double Metrics::MSE(int skipBorderWidth)
{
	if (channels == 1)
	{
		double e = 0.0;
		double wh = width*height;

		for (int i=skipBorderWidth; i<(int)height-skipBorderWidth;i++)
		  for (int j=skipBorderWidth; j<(int)width-skipBorderWidth;j++)
				e += (etalonImg[width * i + j] - testImg[width * i + j]) * (etalonImg[width *i + j] - testImg[width * i + j]);
	
		return e / wh;
	}
	MSEResult result = MseByComponents(skipBorderWidth);
	return (result.R + result.G + result.B) / 3.0;
}

double Metrics::PSNR (int skipBorderWidth)
{
	double mse = MSE(skipBorderWidth);
	if (mse == 0)
		return 1000.0;
	double psnr = 10 * log10(255.0 * 255.0 / mse);
	return psnr;
}

void Metrics::receiveColor(double *out1, double *out2, int colorOffset)
{
	for (unsigned i = 0; i < height; i++)
	{
		for (unsigned j = 0; j < width; j++)
		{
			out1[i * width + j] = etalonImg[(i * width + j) * channels + colorOffset];
			out2[i * width + j] = testImg  [(i * width + j) * channels + colorOffset];
		}
	}
}

double Metrics::SSIM(unsigned long offset, unsigned long interleave)
{
	double *img1 = (double *)malloc(width * height * sizeof(double));
	double *img2 = (double *)malloc(width * height * sizeof(double));
	
	receiveColor(img1, img2, 0);
	double result = ssimFactor(img1, img2, width, height, 255.0, false, offset, interleave);

	if (channels > 1)
	{
		receiveColor(img1, img2, 1);
		result += ssimFactor(img1, img2, width, height, 255.0, false, offset, interleave);
	}

	if (channels > 2)
	{
		receiveColor(img1, img2, 2);
		result += ssimFactor(img1, img2, width, height, 255.0, false, offset, interleave);
	}
	
	free(img1);
	free(img2);

	return result / channels;
}


void Metrics::DiffGistogram (std::vector<int> &res, int skipBorderWidth)
{
	int w3 = width*channels;
	double wh = width*height;
	int max = 0;
	for (int i=skipBorderWidth; i<(int)height-skipBorderWidth;i++)
	  for (int j=skipBorderWidth; j<(int)width-skipBorderWidth;j++)
		{
			if (abs (etalonImg[w3 * i + j * channels] - testImg[w3 * i + j * channels])>max)
				max = abs (etalonImg[w3 * i + j * channels] - testImg[w3 * i + j * channels]);
			if (abs (etalonImg[w3 * i + j * channels + 1] - testImg[w3 * i + j * channels + 1])>max)
				max = abs (etalonImg[w3 * i + j * channels + 1] - testImg[w3 * i + j * channels + 1]);
			if (abs (etalonImg[w3 * i + j * channels + 2] - testImg[w3 * i + j * channels + 2])>max)
				max = abs (etalonImg[w3 * i + j * channels + 2] - testImg[w3 * i + j * channels + 2]);
		}

	for (int k=1;k<=max;k++)
	{
		int cnt = 0;
		for (int i=skipBorderWidth; i<(int)height-skipBorderWidth;i++)
		  for (int j=skipBorderWidth; j<(int)width-skipBorderWidth;j++)
			{
				if (abs (etalonImg[w3 * i + j * channels] - testImg[w3 * i + j * channels])==k)
					cnt++;
				if (abs (etalonImg[w3 * i + j * channels + 1] - testImg[w3 * i + j * channels + 1])==k)
					cnt++;
				if (abs (etalonImg[w3 * i + j * channels + 2] - testImg[w3 * i + j * channels + 2])==k)
					cnt++;
			}
		  res.push_back(cnt);
	}
}


void  Metrics::GetPixelDiff(unsigned char *diff, int skipBorderWidth)
{
	int w3 = width*channels;
	double wh = width*height;
	memset(diff, 0, w3*height);

	for (int i=skipBorderWidth; i<(int)height-skipBorderWidth;i++)
	  for (int j=skipBorderWidth; j<(int)width-skipBorderWidth;j++)
		{
				diff[w3 * i + j * channels] = abs (etalonImg[w3 * i + j * channels] - testImg[w3 * i + j * channels]);
				diff[w3 * i + j * channels + 1]  = abs (etalonImg[w3 * i + j * channels + 1] - testImg[w3 * i + j * channels + 1]);
				diff[w3 * i + j * channels + 2]  = abs (etalonImg[w3 * i + j * channels + 2] - testImg[w3 * i + j * channels + 2]);
		}
}