/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "cube.h"
#include <iostream>
#include <fstream>
#include <sstream>

string CubeLUT::ReadLine(
	ifstream &infile,
	char lineSeparator
) {
	// Skip empty lines and comments
	string textLine("");
	while (textLine.empty() || textLine[0] == '#') {
		if (infile.eof()) {
			status = PrematureEndOfFile;
			break;
		}
		getline(infile, textLine, lineSeparator);
		if (infile.fail()) {
			status = ReadError;
			break;
		}
	}
	return textLine;
} // ReadLine

vector<float> CubeLUT::ParseTableRow(const string &lineOfText) {
	int N = 3;
	tableRow f(N);
	istringstream line(lineOfText);
	for (int i = 0; i < N; i++) {
		line >> f[i];
		if (line.fail()) {
			status = CouldNotParseTableData;
			break;
		};
	}
	return f;
} // ParseTableRow

CubeLUT::LUTState CubeLUT::LoadCubeFile(ifstream &infile) {
	// Set defaults
	status = OK;
	title.clear();
	domainMin = tableRow(3, 0.0);
	domainMax = tableRow(3, 1.0);
	LUT1D.clear();
	LUT3D.clear();
	// Read file data line by line
	const char NewlineCharacter = '\n';
	char lineSeparator = NewlineCharacter;
	// sniff use of legacy lineSeparator
	const char CarriageReturnCharacter = '\r';
	for (int i = 0; i < 255; i++) {
		char inc = infile.get();
		if (inc == NewlineCharacter)
			break;
		if (inc == CarriageReturnCharacter) {
			if (infile.get() == NewlineCharacter)
				break;
			lineSeparator = CarriageReturnCharacter;
			clog << "INFO: This file uses non-compliant line separator\\r(0x0D)" << endl;
			break;
		}
		if (i > 250) {
			status = LineError;
			break;
		}
	}

	infile.seekg(0);
	infile.clear();
	// read keywords
	unsigned N, CntTitle, CntSize, CntMin, CntMax;

	// each keyword to occur zero or one time
	N = CntTitle = CntSize = CntMin = CntMax = 0;
	while (status == OK) {
		std::streampos linePos = infile.tellg();
		string lineOfText = ReadLine(infile, lineSeparator);
		if (!status == OK)
			break;
		// Parse keywords and parameters
		istringstream line(lineOfText);
		string keyword;
		line >> keyword;
		if ("+" < keyword && keyword < ":") {
			// lines of table data come after keywords
			// restore stream pos to re-read line of data
			infile.seekg(linePos);
			break;
		} else if (keyword == "TITLE" && CntTitle++ == 0) {
			const char QUOTE = '"';
			char startOfTitle;
			line >> startOfTitle;
			if (startOfTitle != QUOTE) {
				status = TitleMissingQuote;
				break;
			}
			getline(line, title, QUOTE);
			// read to "
		} else if (keyword == "DOMAIN_MIN" && CntMin++ == 0) {
			line >> domainMin[0] >> domainMin[1] >> domainMin[2];
		} else if (keyword == "DOMAIN_MAX" && CntMax++ == 0) {
			line >> domainMax[0] >> domainMax[1] >> domainMax[2];
		} else if (keyword == "LUT_1D_SIZE" && CntSize++ == 0) {
			line >> N;
			if (N < 2 || N > 65536) {
				status = LUTSizeOutOfRange;
				break;
			}
			LUT1D = table1D(N, tableRow(3));
		} else if (keyword == "LUT_3D_SIZE" && CntSize++ == 0) {
			line >> N;
			if (N < 2 || N > 256) {
				status = LUTSizeOutOfRange;
				break;
			}
			LUT3D = table3D(N, table2D(N, table1D(N, tableRow(3))));
		} else {
			status = UnknownOrRepeatedKeyword;
			break;
		}
		if (line.fail()) {
			status = ReadError;
			break;
		}
	}
	// read keywords
	if (status == OK && CntSize == 0) status = LUTSizeOutOfRange;
	if (status == OK && (domainMin[0] >= domainMax[0] || domainMin[1] >= domainMax[1] || domainMin[2] >= domainMax[2]))
		status = DomainBoundsReversed;
	// read lines of table data
	if (!LUT1D.empty()) {
		N = static_cast<unsigned>(LUT1D.size());
		for (unsigned i = 0; i < N && status == OK; i++) {
			LUT1D[i] = ParseTableRow(ReadLine(infile, lineSeparator));
		}
	} else {
		N = static_cast<unsigned>(LUT3D.size());
		// NOTE that r loops fastest
		for (unsigned b = 0; b < N && status == OK; b++) {
			for (unsigned g = 0; g < N && status == OK; g++) {
				for (unsigned r = 0; r < N && status == OK; r++) {
					LUT3D[r][g][b] = ParseTableRow(ReadLine(infile, lineSeparator));
				}
			}
		}
	}
	// read 3D LUT
	return status;
} // LoadCubeFile

CubeLUT::LUTState CubeLUT::SaveCubeFile(ofstream & outfile) {
	if (!status == OK)
		return status;
	// Write only good Cubes
	// Write keywords
	const char SPACE = ' ';
	const char QUOTE = '"';
	if (!title.empty()) outfile << "TITLE" << SPACE << QUOTE << title << QUOTE << endl;
	outfile << "# Created by CubeLUT.cpp" << endl;
	outfile << "DOMAIN_MIN" << SPACE << domainMin[0] << SPACE << domainMin[1] << SPACE << domainMin[2] << endl;
	outfile << "DOMAIN_MAX" << SPACE << domainMax[0] << SPACE << domainMax[1] << SPACE << domainMax[2] << endl;
	// Write LUT data
	if (!LUT1D.empty()) {
		unsigned N = static_cast<unsigned>(LUT1D.size());
		outfile << "LUT_1D_SIZE" << SPACE << N << endl;
		for (unsigned i = 0; i < N && outfile.good(); i++) {
			outfile << LUT1D[i][0] << SPACE << LUT1D[i][1] << SPACE << LUT1D[i][2] << endl;
		}
	} else {
		unsigned N = static_cast<unsigned>(LUT3D.size());
		outfile << "LUT_3D_SIZE" << SPACE << N << endl;
		// NOTE that r loops fastest
		for (unsigned b = 0; b < N && outfile.good(); b++) {
			for (unsigned g = 0; g < N && outfile.good(); g++) {
				for (unsigned r = 0; r < N && outfile.good(); r++) {
					outfile << LUT3D[r][g][b][0] << SPACE << LUT3D[r][g][b][1] << SPACE << LUT3D[r][g][b][2] << endl;
				}
			}
		}
	}
	// write 3D LUT
	outfile.flush();
	return (outfile.good() ? OK : WriteError);
} // SaveCubeFile
