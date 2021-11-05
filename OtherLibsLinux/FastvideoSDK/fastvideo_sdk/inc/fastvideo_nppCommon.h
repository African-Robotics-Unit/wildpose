/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __FASTVIDEO_NPP_COMMON__
#define __FASTVIDEO_NPP_COMMON__

#include <stdlib.h>
#include "fastvideo_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __GNUC__

#ifdef FAST_EXPORTS
#define DLL __declspec(dllexport) __cdecl
#else
#define DLL
#endif

#else

#define DLL

#endif

typedef enum {
	NPP_INTER_LINEAR,
	NPP_INTER_CUBIC,
	NPP_INTER_CUBIC2P_BSPLINE,
	NPP_INTER_CUBIC2P_CATMULLROM,
	NPP_INTER_CUBIC2P_B05C03,
	NPP_INTER_SUPER,
	NPP_INTER_LANCZOS
} fastNPPImageInterpolation_t;

typedef struct {
	double x;
	double y;
} Point_t;

#ifdef __cplusplus
}
#endif

#endif // __FASTVIDEO_NPP_COMMON__