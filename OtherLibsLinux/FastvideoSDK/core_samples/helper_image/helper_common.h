/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef HELPER_COMMON_H
#define HELPER_COMMON_H

#include <algorithm>
#include <assert.h>

// namespace unnamed (internal)
namespace
{
    //! size of PGM file header
    const unsigned int PGMHeaderSize = 0x40;

    // types

    //! Data converter from unsigned char / unsigned byte to type T
    template<class T>
    struct ConverterFromUByte;

    //! Data converter from unsigned char / unsigned byte
    template<>
    struct ConverterFromUByte<unsigned char>
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        float operator()(const unsigned char &val)
        {
            return static_cast<unsigned char>(val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to float
    template<>
    struct ConverterFromUByte<float>
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        float operator()(const unsigned char &val)
        {
            return static_cast<float>(val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to type T
    template<class T>
    struct ConverterToUByte;

    //! Data converter from unsigned char / unsigned byte to unsigned int
    template<>
    struct ConverterToUByte<unsigned char>
    {
        //! Conversion operator (essentially a passthru
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()(const unsigned char &val)
        {
            return val;
        }
    };

    //! Data converter from unsigned char / unsigned byte to unsigned int
    template<>
    struct ConverterToUByte<float>
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()(const float &val)
        {
		    return static_cast<unsigned char>(val < 0.0f ? 0 : ( val > 255.0f ? 255 : val ));
        }
    };
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result != 0)
#endif
#ifndef SSCANF
#define SSCANF sscanf_s
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result == NULL)
#endif
#ifndef SSCANF
#define SSCANF sscanf
#endif
#endif

template<typename T> static inline T _uSnapUp(T a, T b){
    return a + (b - a % b) % b;
}

#endif