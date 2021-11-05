/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "AffineOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>
#include <string.h>

bool AffineOptions::AffineTypeParser( const char *affineType ) {
	if (strcmp(affineType, "nop") == 0) {
		AffineType = FAST_AFFINE_NOP;
	} else if ( strcmp( affineType, "flip" ) == 0 ) {
		AffineType = FAST_AFFINE_FLIP;
	} else if ( strcmp( affineType, "flop" ) == 0 ) {
		AffineType = FAST_AFFINE_FLOP;
	} else if ( strcmp( affineType, "rotate180" ) == 0 ) {
		AffineType = FAST_AFFINE_ROTATION180;
	} else if ( strcmp( affineType, "rotate90left" ) == 0 ) {
		AffineType = FAST_AFFINE_ROTATION90LEFT;
	} else if (strcmp(affineType, "rotate90left_f") == 0) {
		AffineType = FAST_AFFINE_ROTATION90LEFT_FLOPPED;
	} else if ( strcmp( affineType, "rotate90right" ) == 0 ) {
		AffineType = FAST_AFFINE_ROTATION90RIGHT;
	} else if (strcmp(affineType, "rotate90right_f") == 0) {
		AffineType = FAST_AFFINE_ROTATION90RIGHT_FLOPPED;
	} else {
		fprintf(stderr, "Incorrect affine type (%s)", affineType);
		return false;
	}

	return true;
}

bool AffineOptions::Parse(int argc, char *argv[]) {
	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "affine", &tmp);
	if ( tmp != NULL ) {
		if ( !AffineTypeParser( tmp ) ) {
			return false;
		}
	} else {
		fprintf(stderr, "-affine parameter was not found\n");
		return false;
	}

	return true;
}
