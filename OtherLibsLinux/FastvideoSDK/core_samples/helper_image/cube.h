/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __CUBE_LUT__
#define __CUBE_LUT__

#include <vector>
#include <string>

using namespace std;

class CubeLUT {
public:
	typedef vector<float> tableRow;
	typedef vector<tableRow> table1D;
	typedef vector<table1D> table2D;
	typedef vector <table2D> table3D;
	enum LUTState {
		OK = 0,
		NotInitialized = 1,
		ReadError = 10,
		WriteError,
		PrematureEndOfFile,
		LineError,
		UnknownOrRepeatedKeyword = 20,
		TitleMissingQuote,
		DomainBoundsReversed,
		LUTSizeOutOfRange,
		CouldNotParseTableData
	};
	LUTState status;
	string title;
	tableRow domainMin;
	tableRow domainMax;
	table1D LUT1D;
	table3D LUT3D;
	CubeLUT(void) {
		status = NotInitialized;
	};
	LUTState LoadCubeFile(ifstream &infile);
	LUTState SaveCubeFile(ofstream &outfile);
private:
	string ReadLine(ifstream &infile, char lineSeparator);
	tableRow ParseTableRow(const string &lineOfText);
};

#endif //__CUBE_LUT__