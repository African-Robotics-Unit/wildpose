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

double *CreateGaussFilter(unsigned long w, unsigned long h)
{
	double *gauss = (double *)malloc(w * h * sizeof(double));

	unsigned long d1 = w/2;
	unsigned long d2 = h/2;
	unsigned long x, y;
	double Norm = 0.0;

	for (x = 0; x < w; x++) {
		for(y = 0; y < h; y++) {
			double g = exp(-(((x - d1) * (x - d1) + (y - d2) * (y - d2))/2.25));
			Norm += g;
			gauss[x + y * w] = g;
		}
	}
	for (x = 0; x < w; x++) {
		for(y = 0; y < h; y++) {
			  gauss[x + y * w] = gauss[x + y * w] / Norm;
		}
	}
	return gauss;
}

float ssimFactor(const double *img1, const double *img2,
			unsigned long width, unsigned long height,
			double scale, bool doluminance,
			unsigned long offset, unsigned long interleave)
{
	const double K1 = 0.01;
	const double K2 = 0.03;

	unsigned long counter = 0;
	double ssimsum = 0.0;

	double *gaussFilter = CreateGaussFilter(11, 11);
	
	unsigned long w = 11;
	unsigned long h = 11;
	
	double C1 = (K1*scale)*(K1*scale);
	double C2 = (K2*scale)*(K2*scale);
	//double C3 = C2/2;	

	for(unsigned long y1 = offset;y1 <= height-h-offset;y1 += interleave){
		for(unsigned long x1 = offset;x1 <= width-w-offset;x1++){
			unsigned long x,y;
			double lumvalue_1 = 0.0, lumvalue_2 = 0.0;
			double con2value_1 = 0.0, con2value_2 = 0.0;
			double corrvalue = 0.0;

			for( y=0;y<h;y++){
				for( x=0;x<w;x++){
					double k1 = img1[x + x1 + (y + y1) * width];
					double k2 = img2[x + x1 + (y + y1) * width];
					double valv = gaussFilter[x + y * w];
					lumvalue_1 += k1*valv;
					lumvalue_2 += k2*valv;
					con2value_1 += k1*k1*valv;
					con2value_2 += k2*k2*valv;
					corrvalue += k1*k2*valv;
				}
			}

			con2value_1 -= lumvalue_1 * lumvalue_1;
			con2value_2 -= lumvalue_2 * lumvalue_2;
			corrvalue -= lumvalue_1 * lumvalue_2;

			if (con2value_1	< 0.0)
				con2value_1 = 0.0;
			if (con2value_2	< 0.0)
				con2value_2 = 0.0;

			double complum = !doluminance ? 1.0 : (2.0 * lumvalue_1 * lumvalue_2 + C1)/(lumvalue_1 * lumvalue_1 + lumvalue_2 * lumvalue_2 + C1);
			
			ssimsum += complum * ( (2.0 * corrvalue + C2) / (con2value_1 + con2value_2 + C2) );
			counter++;
		}
	}

	free(gaussFilter);

	if (counter > 0)
		return ssimsum / counter;
	return 1.0;
}
