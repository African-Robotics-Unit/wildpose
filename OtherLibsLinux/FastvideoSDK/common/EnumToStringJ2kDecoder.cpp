/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "EnumToStringJ2kDecoder.h"

const char* EnumToString(J2kCapability_t value) {
	switch (value) {
		case JPEG2000_CAPABILITY_ANY:
			return "JPEG2000_CAPABILITY_ANY";
		case JPEG2000_CAPABILITY_CSTREAM_RESTRICTION_0:
			return "JPEG2000_CAPABILITY_CSTREAM_RESTRICTION_0";
		case JPEG2000_CAPABILITY_CSTREAM_RESTRICTION_1:
			return "JPEG2000_CAPABILITY_CSTREAM_RESTRICTION_1";
		case JPEG2000_CAPABILITY_DCINEMA_2K:
			return "JPEG2000_CAPABILITY_DCINEMA_2K";
		case JPEG2000_CAPABILITY_DCINEMA_4K:
			return "JPEG2000_CAPABILITY_DCINEMA_4K";
		case JPEG2000_CAPABILITY_SCALABLE_DCINEMA_2K:
			return "JPEG2000_CAPABILITY_SCALABLE_DCINEMA_2K";
		case JPEG2000_CAPABILITY_SCALABLE_DCINEMA_4K:
			return "JPEG2000_CAPABILITY_SCALABLE_DCINEMA_4K";
		case JPEG2000_CAPABILITY_OTHER:
			return "JPEG2000_CAPABILITY_OTHER";
		default:
			return "Other";
	}
}

const char* EnumToString(WaveletType wt)
{
	switch (wt)
	{
		case WT_CDF97: return "CDF97";
		case WT_CDF53: return "CDF53";
		case WT_CUSTOM: return "Custom";
		default: return "Unknown";
	}
}

const char* EnumToString(MCT_Type mct) {

	switch (mct)
	{
		case MCT_None: return "No";
		case MCT_Reversible: return "Reversible";
		case MCT_Irreversible: return "Irreversible";
		default: return "Unknown";
	}
}

const char* EnumToString(ProgressionType prg) {
	switch (prg)
	{
		case PT_LRCP: return  "LRCP";
		case PT_RLCP: return "RLCP";
		case PT_RPCL: return "RPCL";
		case PT_PCRL: return "PCRL";
		case PT_CPRL: return "CPRL";
		default: return "Unknown";
	}
}
