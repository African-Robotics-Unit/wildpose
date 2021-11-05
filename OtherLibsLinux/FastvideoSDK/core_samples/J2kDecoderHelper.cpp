/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "J2kDecoderHelper.h"

#include <cstring>
#include <fstream>

void FreeJ2kImageInfo(fastJ2kImageInfo_t* info)
{
    delete [] info->components;

    unsigned i, j;
    for (i = 0; i < info->asocBoxesCount; i++)
    {
        for (j = 0; j < info->asoc[i].labelCount; j++)
            delete [] info->asoc[i].labels[j];
        if (info->asoc[i].labelCount > 0)
        {
            delete [] info->asoc[i].labels;
            delete [] info->asoc[i].labelLengths;
        }

        for (j = 0; j < info->asoc[i].xmlCount; j++)
            delete [] info->asoc[i].XMLs[j];
        if (info->asoc[i].xmlCount > 0)
        {
            delete [] info->asoc[i].XMLs;
            delete [] info->asoc[i].xmlLengths;
        }
        if (info->asoc[i].childrenCount > 0)
            delete [] info->asoc[i].children;
    }

    if (info->asocBoxesCount > 0)
        delete [] info->asoc;

    if (info->containsRreqBox)
    {
        if (info->rreq.standardFlagsCount > 0)
        {
            delete [] info->rreq.standardFlags;
            delete [] info->rreq.standardMasks;
        }
        if (info->rreq.vendorFeatureCount > 0)
        {
            delete [] info->rreq.vendorFeatures;
            delete [] info->rreq.vendorMasks;
        }
    }

    if (info->uuidBoxesCount > 0)
    {
        for (i = 0; i < info->uuidBoxesCount; i++)
        {
            if (info->uuidBoxes[i].dataLength > 0)
                delete [] info->uuidBoxes[i].data;
        }
        delete [] info->uuidBoxes;
    }

    if (info->containsUuidInfoBox)
    {
        if (info->uuidInfo.idCount > 0)
            delete [] info->uuidInfo.IDs;
        if (info->uuidInfo.urlLength > 0)
            delete [] info->uuidInfo.url;
    }
}

int WriteJ2kGML(const char* filename, fastJ2kImageInfo_t* info)
{
    const char *label = "gml.root-instance";
    size_t labelLength = strlen(label);
    for (unsigned i = 0; i < info->asocBoxesCount; i++)
    {
        if (info->asoc[i].labelCount > 0 && strncmp(reinterpret_cast<const char*>(info->asoc[i].labels[0]), label, labelLength) == 0)
        {
            std::fstream fh(filename, std::fstream::out);
            if (fh.bad()) return 0;

            fh << info->asoc[i].XMLs[0];

            fh.flush();
            if (fh.bad()) return 0;

            fh.close();
            return 1;
        }
    }
    return 0;
}
