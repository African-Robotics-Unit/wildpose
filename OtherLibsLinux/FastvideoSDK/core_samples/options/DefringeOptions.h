/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __DEFRINGE_OPTIONS__
#define __DEFRINGE_OPTIONS__

#include "fastvideo_sdk.h"

class DefringeOptions {
public:
	unsigned WindowSize;

	unsigned short TintR;
	unsigned short TintG;
	unsigned short TintB;

	float Fi_tint; // ����, ������������ �������: [-180, +180]
	float Fi_max; // ������������ ���� ���������� �� ��������� �������: [60, 90]
	float Coefficient; // [0.001, 0.1]

	DefringeOptions(void) { };
	~DefringeOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __DEFRINGE_OPTIONS__