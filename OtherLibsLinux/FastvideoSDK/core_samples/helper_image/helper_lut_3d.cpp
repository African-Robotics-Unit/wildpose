/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include <fstream>
#include <memory>
#include <string.h>

#include "fastvideo_sdk.h"
#include "FastAllocator.h"
#include "helper_common.h"
#include "helper_lut_3d.hpp"

#include "cube.h"
#include "PugiXml/pugixml.hpp"

fastStatus_t fvLoadCube3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_r,
	std::unique_ptr<float, FastAllocator> &data_g,
	std::unique_ptr<float, FastAllocator> &data_b,
	unsigned &size
) {
	ifstream infile(file);
	if (!infile.good()) {
		fprintf(stderr, "Could not open input file %s\n", file);
		return FAST_IO_ERROR;
	}

	CubeLUT cube;
	const CubeLUT::LUTState ret = cube.LoadCubeFile(infile);
	infile.close();
	if (ret != CubeLUT::LUTState::OK) {
		fprintf(stderr, "Could not parse the cube info in the input file\n");
		return FAST_IO_ERROR;
	}

	size = static_cast<unsigned>(cube.LUT3D.size());
	FastAllocator alloc;
	data_r.reset((float *)alloc.allocate(size * size * size * sizeof(float)));
	data_g.reset((float *)alloc.allocate(size * size * size * sizeof(float)));
	data_b.reset((float *)alloc.allocate(size * size * size * sizeof(float)));

	for (unsigned b = 0; b < size; b++) {
		for (unsigned g = 0; g < size; g++) {
			for (unsigned r = 0; r < size; r++) {
				CubeLUT::tableRow row = cube.LUT3D[r][g][b];

				data_r.get()[b * size * size + g * size + r] = row[0];
				data_g.get()[b * size * size + g * size + r] = row[1];
				data_b.get()[b * size * size + g * size + r] = row[2];
			}
		}
	}

	return FAST_OK;
}

fastStatus_t fvSaveCube3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_r,
	std::unique_ptr<float, FastAllocator> &data_g,
	std::unique_ptr<float, FastAllocator> &data_b,
	unsigned size,
	unsigned bitsPerChannel
) {
	ofstream ofile(file);
	if (!ofile.good()) {
		fprintf(stderr, "Could not open input file %s\n", file);
		return FAST_IO_ERROR;
	}

	CubeLUT cube;
	{
		cube.title = "Default LUT 3D table";
		cube.domainMin = CubeLUT::tableRow(3, 0.0);
		cube.domainMax = CubeLUT::tableRow(3, (1 << bitsPerChannel) - 1);
		cube.LUT1D.clear();
		cube.LUT3D = CubeLUT::table3D(size, CubeLUT::table2D(size, CubeLUT::table1D(size, CubeLUT::tableRow(3))));
		cube.status = CubeLUT::OK;
	}

	for (unsigned b = 0; b < size; b++) {
		for (unsigned g = 0; g < size; g++) {
			for (unsigned r = 0; r < size; r++) {
				CubeLUT::tableRow row(3);
				{
					row[0] = data_r.get()[b * size * size + g * size + r];
					row[1] = data_g.get()[b * size * size + g * size + r];
					row[2] = data_b.get()[b * size * size + g * size + r];
				}
				cube.LUT3D[r][g][b] = row;
			}
		}
	}

	const CubeLUT::LUTState ret = cube.SaveCubeFile(ofile);
	ofile.close();
	if (ret != CubeLUT::LUTState::OK) {
		fprintf(stderr, "Could not parse the cube info in the input file\n");
		return FAST_IO_ERROR;
	}

	return FAST_OK;
}

struct allow_remote_predicate {
	bool operator()(pugi::xml_attribute attr) const {
		return strcmp(attr.name(), "hueDivisions") == 0;
	}

	bool operator()(pugi::xml_node node) const {
		return node.attribute("hueDivisions") != NULL;
	}
};

fastStatus_t fvLoadXml3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_h,
	std::unique_ptr<float, FastAllocator> &data_s,
	std::unique_ptr<float, FastAllocator> &data_v,
	fast_uint3* size,
	fastColorSaturationOperationType_t operationType[3]
) {
	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file(file);
	if (!result) {
		fprintf(stderr, "Parsing error (%s): %s\n", file, result.description());
		return FAST_IO_ERROR;
	}

	pugi::xml_node root = doc.document_element();
	pugi::xml_node nodes = root.find_child(allow_remote_predicate());
	if (nodes == NULL) {
		fprintf(stderr, "Parsing error\n");
		return FAST_IO_ERROR;
	}
	{
		size->x = nodes.attribute("hueDivisions").as_int();
		size->y = nodes.attribute("satDivisions").as_int();
		size->z = nodes.attribute("valDivisions").as_int();

		FastAllocator alloc;
		data_h.reset((float *)alloc.allocate(size->x * size->y * size->z * sizeof(float)));
		data_s.reset((float *)alloc.allocate(size->x * size->y * size->z * sizeof(float)));
		data_v.reset((float *)alloc.allocate(size->x * size->y * size->z * sizeof(float)));
	}

	pugi::char_t *hue, *sat, *val;
	{
		pugi::xml_node node = nodes.child("Element");
		operationType[0] = node.attribute("HueShift") == NULL ? FAST_OP_MULTIPLY : FAST_OP_ADD;
		operationType[1] = node.attribute("SatShift") == NULL ? FAST_OP_MULTIPLY : FAST_OP_ADD;
		operationType[2] = node.attribute("ValShift") == NULL ? FAST_OP_MULTIPLY : FAST_OP_ADD;

		hue = node.attribute("HueShift") == NULL ? (pugi::char_t *)"HueScale" : (pugi::char_t *)"HueShift";
		sat = node.attribute("SatShift") == NULL ? (pugi::char_t *)"SatScale" : (pugi::char_t *)"SatShift";
		val = node.attribute("ValShift") == NULL ? (pugi::char_t *)"ValScale" : (pugi::char_t *)"ValShift";
	}

	unsigned idx = 0;
	float *ptr_h = data_h.get();
	float *ptr_s = data_s.get();
	float *ptr_v = data_v.get();
	for (pugi::xml_node node = nodes.child("Element"); node; node = node.next_sibling("Element")) {
		ptr_h[idx] = node.attribute(hue).as_float();
		ptr_s[idx] = node.attribute(sat).as_float();
		ptr_v[idx] = node.attribute(val).as_float();

		idx++;
	}

	if ((size->x * size->y * size->z) != idx) {
		fprintf(stderr, "FAILED. Find %d elements in file\n", idx);
		return FAST_IO_ERROR;
	}

	return FAST_OK;
}

fastStatus_t fvSaveXml3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_h,
	std::unique_ptr<float, FastAllocator> &data_s,
	std::unique_ptr<float, FastAllocator> &data_v,
	fast_uint3 &size,
	fastColorSaturationOperationType_t operationType[3]
) {
	pugi::xml_document doc;
	doc.load_string("<dcpData><ProfileName>Test system (extended smem)</ProfileName></dcpData>");

	pugi::xml_node decl = doc.prepend_child(pugi::node_declaration);
	decl.append_attribute("version") = "1.0";
	decl.append_attribute("encoding") = "UTF-8";

	pugi::xml_node hsvTable = doc.child("dcpData").append_child("HueSatTable");
	{
		hsvTable.append_attribute("hueDivisions") = size.x;
		hsvTable.append_attribute("satDivisions") = size.y;
		hsvTable.append_attribute("valDivisions") = size.z;
	}

	pugi::char_t *hue = operationType[0] == FAST_OP_MULTIPLY ? (pugi::char_t *)"HueScale" : (pugi::char_t *)"HueShift";
	pugi::char_t *sat = operationType[1] == FAST_OP_MULTIPLY ? (pugi::char_t *)"SatScale" : (pugi::char_t *)"SatShift";
	pugi::char_t *val = operationType[2] == FAST_OP_MULTIPLY ? (pugi::char_t *)"ValScale" : (pugi::char_t *)"ValShift";

	float *ptr_h = data_h.get();
	float *ptr_s = data_s.get();
	float *ptr_v = data_v.get();
	for (unsigned h = 0; h < size.x; h++) {
		for (unsigned s = 0; s < size.y; s++) {
			for (unsigned v = 0; v < size.z; v++) {
				pugi::xml_node node = hsvTable.append_child("Element");

				node.append_attribute("hueDiv") = h;
				node.append_attribute("satDiv") = s;
				node.append_attribute("valDiv") = v;

				char tmp[50];
				sprintf(tmp, "%.06f", ptr_h[h * size.y * size.z + s * size.z + v]);
				node.append_attribute(hue) = tmp;
				sprintf(tmp, "%.06f", ptr_s[h * size.y * size.z + s * size.z + v]);
				node.append_attribute(sat) = tmp;
				sprintf(tmp, "%.06f", ptr_v[h * size.y * size.z + s * size.z + v]);
				node.append_attribute(val) = tmp;
			}
		}
	}

	doc.save_file(file);

	return FAST_OK;
}
